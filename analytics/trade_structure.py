"""
analytics/trade_structure.py
==============================
Trade Structure Engine for Short Volatility via Correlation/Dispersion

Constructs actionable trade tickets from SignalStackResult objects.
Three trade archetypes:

  1. **Sector Relative-Value** (existing)
     Long/Short sector ETF vs SPY — beta-neutral, PCA residual driven.

  2. **Dispersion Trade** (new)
     Short index vol + Long constituent vol — profits from realized
     dispersion exceeding implied. Constructed as:
       - Short SPY straddle/strangle (vega-neutral notional)
       - Long sector ETF straddles (weighted by sector weight × vega ratio)
     PnL ≈ Σ(w_i² σ_i²) - σ_index²  (realized disp. vs. implied)

  3. **RV Spread** (new)
     Pair trade between two sectors:
       - Long dislocated sector, Short rich sector
       - Hedge-ratio from cointegration regression
       - Mean-reversion driven

Each trade includes:
  - Leg construction (instrument, direction, notional, hedge ratio)
  - Greeks profile (delta, vega, gamma, theta proxy)
  - Risk limits (max loss, stop-loss levels)
  - Exit conditions (z-score compression, time, regime change)

Ref: Dispersion P&L decomposition (Jacquier & Slaoui)
Ref: Cboe Implied Correlation Index
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
# Trade Leg
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeLeg:
    """Single leg of a trade."""
    instrument: str          # e.g., "XLK", "SPY", "XLK straddle"
    direction: str           # "BUY" / "SELL"
    notional_weight: float   # Weight relative to portfolio (0-1)
    hedge_ratio: float       # h in log-spread: x = ln(A) - h·ln(B)
    instrument_type: str     # "equity" / "straddle" / "strangle" / "put_spread" / "call_spread"
    expiry_target_days: int  # Target DTE for options legs (0 = equity)

    @property
    def description(self) -> str:
        if self.instrument_type == "equity":
            return f"{self.direction} {self.instrument} notional={self.notional_weight:.4f}"
        return (
            f"{self.direction} {self.instrument} {self.instrument_type} "
            f"notional={self.notional_weight:.4f} DTE~{self.expiry_target_days}d"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Greeks Profile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GreeksProfile:
    """Synthetic Greeks for a trade structure."""
    delta_spy: float         # Net SPY beta exposure
    delta_tnx: float         # Rate sensitivity
    delta_dxy: float         # Dollar sensitivity
    vega_net: float          # Net vega (positive = long vol)
    gamma_net: float         # Net gamma
    theta_daily: float       # Estimated daily theta (negative = pay)
    rho_corr: float          # Sensitivity to correlation changes
    rho_dispersion: float    # Sensitivity to dispersion changes

    @property
    def is_short_vol(self) -> bool:
        return self.vega_net < -0.01

    @property
    def is_long_dispersion(self) -> bool:
        return self.rho_dispersion > 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Exit Conditions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExitConditions:
    """When to exit the trade."""
    # Z-score based
    z_entry: float               # Z at entry
    z_target: float              # Target z for profit-take (typically 0 or small)
    z_stop: float                # Stop-loss z (further from mean)

    # Time based
    max_holding_days: int        # Hard time limit
    half_life_est: float         # Expected HL for time decay monitoring

    # Regime based
    regime_kill_states: List[str]  # Kill if regime enters these states
    safety_floor: float          # Kill if S^safe drops below this

    # P&L based
    max_loss_pct: float          # Max loss as % of notional
    profit_target_pct: float     # Profit target as % of notional

    @property
    def description(self) -> str:
        return (
            f"Exit: z→{self.z_target:.1f} (target), z→{self.z_stop:.1f} (stop), "
            f"max {self.max_holding_days}d, loss<{self.max_loss_pct:.1%}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Trade Ticket (complete structure)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeTicket:
    """Complete trade structure ready for execution."""
    trade_id: str                # Unique identifier
    trade_type: str              # "sector_rv" / "dispersion" / "rv_spread"
    direction: str               # "LONG" / "SHORT" (primary direction)
    ticker: str                  # Primary identifier

    # From signal stack
    conviction_score: float
    distortion_score: float
    dislocation_score: float
    mean_reversion_score: float
    regime_safety_score: float

    # Sizing
    raw_weight: float            # Pre-regime weight
    final_weight: float          # Post-regime/risk-adjusted weight
    size_multiplier: float       # From regime safety

    # Structure
    legs: List[TradeLeg]
    greeks: GreeksProfile
    exit_conditions: ExitConditions

    # Metadata
    entry_z: float
    entry_residual: float
    half_life_est: float
    pm_note: str                 # Human-readable rationale

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def is_active(self) -> bool:
        return self.final_weight > 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Trade Construction Engine
# ─────────────────────────────────────────────────────────────────────────────

class TradeStructureEngine:
    """
    Converts SignalStackResult objects into executable TradeTicket objects.

    Integrates with:
    - SignalStackEngine (Layer 1-4 scores)
    - QuantEngine (betas, vol, hedge ratios)
    - RegimeSafetyResult (sizing caps)
    """

    def __init__(self, settings=None):
        self.settings = settings

        # Default DTE targets for options legs
        self.dte_short_index = getattr(settings, "trade_dte_short_index", 30) if settings else 30
        self.dte_long_sector = getattr(settings, "trade_dte_long_sector", 45) if settings else 45

        # Risk limits
        self.max_loss_pct = getattr(settings, "trade_max_loss_pct", 0.02) if settings else 0.02
        self.profit_target_pct = getattr(settings, "trade_profit_target_pct", 0.015) if settings else 0.015
        self.max_holding_days = getattr(settings, "trade_max_holding_days", 45) if settings else 45

        # Z-score exit parameters
        self.z_target_ratio = 0.20  # Exit when z compresses to 20% of entry
        self.z_stop_ratio = 1.50    # Stop if z extends 50% beyond entry

        # Concentration limits
        self.max_single_trade_weight = getattr(settings, "max_single_name_weight", 0.20) if settings else 0.20
        self.max_gross_dispersion = getattr(settings, "trade_max_gross_dispersion", 0.40) if settings else 0.40

    # ─── Sector RV Trade ─────────────────────────────────────────────

    def construct_sector_rv(
        self,
        signal_result,
        *,
        beta_spy: float = 1.0,
        ewma_vol: float = 0.20,
        half_life: float = float("nan"),
        beta_tnx: float = 0.0,
        beta_dxy: float = 0.0,
    ) -> TradeTicket:
        """
        Construct sector relative-value trade: Long/Short sector vs SPY.

        Parameters
        ----------
        signal_result : SignalStackResult from signal stack
        beta_spy      : Sector beta to SPY (for hedge ratio)
        ewma_vol      : Sector annualized EWMA volatility
        half_life     : Estimated mean-reversion half-life
        """
        ticker = signal_result.ticker
        direction = signal_result.direction
        conv = signal_result.conviction_score
        z = signal_result.residual_z

        # Raw weight: conviction-proportional
        raw_w = min(self.max_single_trade_weight, conv * 0.25)

        # Regime-adjusted
        final_w = raw_w * signal_result.size_multiplier

        # Hedge ratio
        h = beta_spy if math.isfinite(beta_spy) else 1.0

        # Legs
        if direction == "LONG":
            legs = [
                TradeLeg(ticker, "BUY", final_w, 1.0, "equity", 0),
                TradeLeg("SPY", "SELL", final_w * h, h, "equity", 0),
            ]
        elif direction == "SHORT":
            legs = [
                TradeLeg(ticker, "SELL", final_w, 1.0, "equity", 0),
                TradeLeg("SPY", "BUY", final_w * h, h, "equity", 0),
            ]
        else:
            legs = []
            final_w = 0.0

        # Greeks
        net_delta_spy = -final_w * h if direction == "LONG" else final_w * h if direction == "SHORT" else 0.0
        # Residual delta should be near zero (beta-hedged)
        greeks = GreeksProfile(
            delta_spy=round(net_delta_spy, 6),
            delta_tnx=round(-final_w * beta_tnx, 6) if direction == "LONG" else round(final_w * beta_tnx, 6),
            delta_dxy=round(-final_w * beta_dxy, 6) if direction == "LONG" else round(final_w * beta_dxy, 6),
            vega_net=0.0,  # Equity trade → no direct vega
            gamma_net=round(abs(z) * final_w, 6),  # Synthetic gamma from z-score
            theta_daily=0.0,  # Equity → no theta
            rho_corr=round(-final_w * 0.3, 6),  # Approximate correlation sensitivity
            rho_dispersion=round(final_w * 0.2, 6),
        )

        # Exit conditions
        hl = half_life if math.isfinite(half_life) else 30.0
        exit_cond = ExitConditions(
            z_entry=z,
            z_target=z * self.z_target_ratio,
            z_stop=z * self.z_stop_ratio if abs(z) > 0.5 else z - 1.5 * np.sign(z),
            max_holding_days=min(self.max_holding_days, int(hl * 3)),
            half_life_est=hl,
            regime_kill_states=["CRISIS"],
            safety_floor=0.10,
            max_loss_pct=self.max_loss_pct,
            profit_target_pct=self.profit_target_pct,
        )

        pm_note = (
            f"{'Buy' if direction == 'LONG' else 'Sell'} {ticker} vs SPY "
            f"(h={h:.2f}). Z={z:+.2f}, conv={conv:.2f}, "
            f"MR={signal_result.mean_reversion_score:.2f}, "
            f"HL~{hl:.0f}d. "
            f"Target z→{exit_cond.z_target:+.1f}, stop z→{exit_cond.z_stop:+.1f}."
        )

        return TradeTicket(
            trade_id=f"SRV_{ticker}_{direction}",
            trade_type="sector_rv",
            direction=direction,
            ticker=ticker,
            conviction_score=conv,
            distortion_score=signal_result.distortion_score,
            dislocation_score=signal_result.dislocation_score,
            mean_reversion_score=signal_result.mean_reversion_score,
            regime_safety_score=signal_result.regime_safety_score,
            raw_weight=round(raw_w, 6),
            final_weight=round(final_w, 6),
            size_multiplier=signal_result.size_multiplier,
            legs=legs,
            greeks=greeks,
            exit_conditions=exit_cond,
            entry_z=z,
            entry_residual=signal_result.dislocation_detail.residual_value if signal_result.dislocation_detail else float("nan"),
            half_life_est=hl,
            pm_note=pm_note,
        )

    # ─── Dispersion Trade ────────────────────────────────────────────

    def construct_dispersion(
        self,
        signal_result,
        sector_weights: Dict[str, float],
        sector_vols: Dict[str, float],
        index_vol: float = 0.15,
        *,
        target_vega: float = 0.10,
    ) -> TradeTicket:
        """
        Construct dispersion trade: Short index vol + Long sector vol.

        The trade profits when realized dispersion exceeds implied:
          PnL ∝ Σ(w_i² σ_i²) - σ_index²

        Parameters
        ----------
        signal_result   : SignalStackResult (should have high distortion)
        sector_weights  : {ticker: weight in index} — typically SPY sector weights
        sector_vols     : {ticker: annualized vol}
        index_vol       : SPY annualized vol
        target_vega     : Target net vega notional
        """
        conv = signal_result.conviction_score
        dist = signal_result.distortion_score

        # Raw weight scaled by distortion (higher distortion = more dispersion opportunity)
        raw_w = min(self.max_gross_dispersion, dist * conv * 0.30)
        final_w = raw_w * signal_result.size_multiplier

        # Short index leg: sell SPY straddle
        legs = [
            TradeLeg(
                "SPY", "SELL", final_w,
                hedge_ratio=1.0,
                instrument_type="straddle",
                expiry_target_days=self.dte_short_index,
            )
        ]

        # Long sector legs: buy sector straddles, weighted by sector weight × vega ratio
        total_sector_vega = 0.0
        for ticker, sw in sector_weights.items():
            if sw < 0.01:
                continue
            sv = sector_vols.get(ticker, index_vol)
            vega_ratio = sv / max(index_vol, 0.01)
            leg_weight = final_w * sw * vega_ratio
            legs.append(TradeLeg(
                ticker, "BUY", round(leg_weight, 6),
                hedge_ratio=sw,
                instrument_type="straddle",
                expiry_target_days=self.dte_long_sector,
            ))
            total_sector_vega += leg_weight

        # Greeks for dispersion
        # Net vega: short index + long sectors ≈ near-zero if properly weighted
        net_vega = total_sector_vega - final_w
        # Theta: short straddle earns theta, long straddles pay → net depends on term structure
        theta_approx = -final_w * index_vol / math.sqrt(252) * 0.5 + total_sector_vega * 0.3 / math.sqrt(252)

        greeks = GreeksProfile(
            delta_spy=0.0,  # Delta-neutral by construction
            delta_tnx=0.0,
            delta_dxy=0.0,
            vega_net=round(net_vega, 6),
            gamma_net=round(final_w * 0.8, 6),  # Long gamma from sector straddles
            theta_daily=round(theta_approx, 6),
            rho_corr=round(-final_w * 0.6, 6),  # Profits when corr drops → negative rho_corr
            rho_dispersion=round(final_w * 0.8, 6),  # Profits when dispersion rises
        )

        z = signal_result.residual_z
        exit_cond = ExitConditions(
            z_entry=z,
            z_target=z * 0.3,
            z_stop=z * 1.5,
            max_holding_days=self.dte_short_index,  # Expire with short leg
            half_life_est=float("nan"),
            regime_kill_states=["CRISIS", "TENSION"],  # Tighter for dispersion
            safety_floor=0.30,  # Higher floor for dispersion
            max_loss_pct=self.max_loss_pct * 1.5,  # Wider stop for dispersion
            profit_target_pct=self.profit_target_pct * 2.0,  # Higher target
        )

        pm_note = (
            f"Dispersion: Short SPY straddle + Long sector straddles. "
            f"Distortion={dist:.2f}, conv={conv:.2f}. "
            f"Net vega={net_vega:+.4f}, {len(legs)-1} sector legs. "
            f"Profits if realized disp > implied. "
            f"Kill on CRISIS/TENSION."
        )

        return TradeTicket(
            trade_id=f"DISP_{signal_result.ticker}",
            trade_type="dispersion",
            direction="LONG_DISPERSION",
            ticker=signal_result.ticker,
            conviction_score=conv,
            distortion_score=dist,
            dislocation_score=signal_result.dislocation_score,
            mean_reversion_score=signal_result.mean_reversion_score,
            regime_safety_score=signal_result.regime_safety_score,
            raw_weight=round(raw_w, 6),
            final_weight=round(final_w, 6),
            size_multiplier=signal_result.size_multiplier,
            legs=legs,
            greeks=greeks,
            exit_conditions=exit_cond,
            entry_z=z,
            entry_residual=float("nan"),
            half_life_est=float("nan"),
            pm_note=pm_note,
        )

    # ─── RV Spread Trade ─────────────────────────────────────────────

    def construct_rv_spread(
        self,
        signal_result,
        hedge_ratio: float = 1.0,
        ticker_a: str = "",
        ticker_b: str = "",
        *,
        vol_a: float = 0.20,
        vol_b: float = 0.20,
        half_life: float = float("nan"),
        beta_a_spy: float = 1.0,
        beta_b_spy: float = 1.0,
    ) -> TradeTicket:
        """
        Construct RV spread: Long sector A + Short sector B (or vice versa).

        Spread = ln(P_A) - h·ln(P_B)
        If z < 0 → LONG spread → BUY A, SELL B
        If z > 0 → SHORT spread → SELL A, BUY B
        """
        conv = signal_result.conviction_score
        z = signal_result.residual_z
        direction = signal_result.direction

        # Parse tickers from spread name if not provided
        if not ticker_a and "-" in signal_result.ticker:
            parts = signal_result.ticker.split("-")
            ticker_a, ticker_b = parts[0], parts[1]

        raw_w = min(self.max_single_trade_weight, conv * 0.20)
        final_w = raw_w * signal_result.size_multiplier

        h = hedge_ratio if math.isfinite(hedge_ratio) else 1.0

        if direction == "LONG":
            # Long spread = BUY A, SELL B
            legs = [
                TradeLeg(ticker_a, "BUY", final_w, 1.0, "equity", 0),
                TradeLeg(ticker_b, "SELL", final_w * h, h, "equity", 0),
            ]
            net_spy = final_w * (beta_a_spy - h * beta_b_spy)
        elif direction == "SHORT":
            legs = [
                TradeLeg(ticker_a, "SELL", final_w, 1.0, "equity", 0),
                TradeLeg(ticker_b, "BUY", final_w * h, h, "equity", 0),
            ]
            net_spy = final_w * (-beta_a_spy + h * beta_b_spy)
        else:
            legs = []
            final_w = 0.0
            net_spy = 0.0

        # Optional SPY hedge if residual beta is significant
        if abs(net_spy) > 0.05 and final_w > 0:
            legs.append(TradeLeg(
                "SPY", "SELL" if net_spy > 0 else "BUY",
                abs(net_spy), abs(net_spy / final_w), "equity", 0,
            ))
            net_spy = 0.0

        greeks = GreeksProfile(
            delta_spy=round(net_spy, 6),
            delta_tnx=0.0,
            delta_dxy=0.0,
            vega_net=0.0,
            gamma_net=round(abs(z) * final_w * 0.5, 6),
            theta_daily=0.0,
            rho_corr=round(-final_w * 0.15, 6),
            rho_dispersion=round(final_w * 0.10, 6),
        )

        hl = half_life if math.isfinite(half_life) else 25.0
        exit_cond = ExitConditions(
            z_entry=z,
            z_target=z * self.z_target_ratio,
            z_stop=z * self.z_stop_ratio if abs(z) > 0.5 else z - 1.5 * np.sign(z),
            max_holding_days=min(self.max_holding_days, int(hl * 3)),
            half_life_est=hl,
            regime_kill_states=["CRISIS"],
            safety_floor=0.10,
            max_loss_pct=self.max_loss_pct,
            profit_target_pct=self.profit_target_pct,
        )

        pm_note = (
            f"RV Spread: {'Long' if direction == 'LONG' else 'Short'} {ticker_a} vs {ticker_b} "
            f"(h={h:.3f}). Z={z:+.2f}, conv={conv:.2f}, "
            f"MR={signal_result.mean_reversion_score:.2f}, "
            f"HL~{hl:.0f}d."
        )

        return TradeTicket(
            trade_id=f"RV_{ticker_a}_{ticker_b}_{direction}",
            trade_type="rv_spread",
            direction=direction,
            ticker=signal_result.ticker,
            conviction_score=conv,
            distortion_score=signal_result.distortion_score,
            dislocation_score=signal_result.dislocation_score,
            mean_reversion_score=signal_result.mean_reversion_score,
            regime_safety_score=signal_result.regime_safety_score,
            raw_weight=round(raw_w, 6),
            final_weight=round(final_w, 6),
            size_multiplier=signal_result.size_multiplier,
            legs=legs,
            greeks=greeks,
            exit_conditions=exit_cond,
            entry_z=z,
            entry_residual=signal_result.dislocation_detail.residual_value if signal_result.dislocation_detail else float("nan"),
            half_life_est=hl,
            pm_note=pm_note,
        )

    # ─── Batch: all passing candidates ────────────────────────────────

    def construct_all_trades(
        self,
        signal_results: list,
        *,
        master_df: Optional[pd.DataFrame] = None,
        sector_weights: Optional[Dict[str, float]] = None,
        sector_vols: Optional[Dict[str, float]] = None,
        index_vol: float = 0.15,
        rv_spreads: Optional[dict] = None,
    ) -> List[TradeTicket]:
        """
        Construct trade tickets for all passing signal results.

        Parameters
        ----------
        signal_results  : List[SignalStackResult] from signal stack
        master_df       : QuantEngine master_df (for betas, vols, HL)
        sector_weights  : {ticker: index_weight} for dispersion
        sector_vols     : {ticker: ann_vol} for dispersion
        rv_spreads      : {name: (series, h, a, b)} for RV spread trades
        """
        tickets = []

        # Build lookup from master_df
        lookup = {}
        if master_df is not None:
            for _, row in master_df.iterrows():
                t = str(row.get("sector_ticker", ""))
                if t:
                    lookup[t] = row

        for sr in signal_results:
            if not sr.passes_entry:
                continue

            if sr.trade_type == "sector":
                # Sector RV trade
                info = lookup.get(sr.ticker, {})
                ticket = self.construct_sector_rv(
                    sr,
                    beta_spy=_safe(info, "beta_spy_60d", 1.0),
                    ewma_vol=_safe(info, "ewma_vol_ann", 0.20),
                    half_life=_safe(info, "half_life_days_est", float("nan")),
                    beta_tnx=_safe(info, "beta_tnx_60d", 0.0),
                    beta_dxy=_safe(info, "beta_dxy_60d", 0.0),
                )
                tickets.append(ticket)

            elif sr.trade_type == "rv_spread":
                # RV spread trade
                parts = sr.ticker.split("-") if "-" in sr.ticker else ["", ""]
                h = 1.0
                if rv_spreads and sr.ticker in rv_spreads:
                    _, h, _, _ = rv_spreads[sr.ticker]
                ticket = self.construct_rv_spread(
                    sr,
                    hedge_ratio=h,
                    ticker_a=parts[0],
                    ticker_b=parts[1],
                    beta_a_spy=_safe(lookup.get(parts[0], {}), "beta_spy_60d", 1.0),
                    beta_b_spy=_safe(lookup.get(parts[1], {}), "beta_spy_60d", 1.0),
                )
                tickets.append(ticket)

        # Optional dispersion trade if distortion is high enough
        if (
            signal_results
            and signal_results[0].distortion_score >= 0.6
            and sector_weights
            and sector_vols
        ):
            disp_ticket = self.construct_dispersion(
                signal_results[0],  # Use top candidate for metadata
                sector_weights=sector_weights,
                sector_vols=sector_vols,
                index_vol=index_vol,
            )
            if disp_ticket.is_active:
                tickets.append(disp_ticket)

        return tickets


# ─────────────────────────────────────────────────────────────────────────────
# Position Sizing Engine
# ─────────────────────────────────────────────────────────────────────────────

class PositionSizingEngine:
    """
    Conviction-based position sizing with regime caps and risk constraints.

    Sizing pipeline:
      1. Base weight from conviction: w_base = f(conviction)
      2. Regime scale: w_regime = w_base × S^safe × size_cap
      3. Vol-targeting: w_vol = w_regime × (target_vol / realized_vol)
      4. Concentration: clip per-name and gross
      5. Correlation adjustment: reduce if portfolio corr is high
    """

    def __init__(self, settings=None):
        self.settings = settings
        self.target_vol = getattr(settings, "target_portfolio_vol", 0.08) if settings else 0.08
        self.max_single = getattr(settings, "max_single_name_weight", 0.20) if settings else 0.20
        self.max_gross = getattr(settings, "max_leverage_calm", 3.0) if settings else 3.0

    def size_portfolio(
        self,
        tickets: List[TradeTicket],
        regime_safety_score: float = 1.0,
        regime_size_cap: float = 1.0,
    ) -> List[TradeTicket]:
        """
        Apply portfolio-level sizing constraints to a set of trade tickets.

        Steps:
          1. Scale all weights to fit gross leverage limit
          2. Apply per-name concentration limit
          3. Regime-scale: multiply by size_cap
          4. Net exposure constraint (max 30% net — target market-neutral)
          5. Max 8 concurrent positions

        Returns tickets with updated final_weight.
        """
        if not tickets:
            return tickets

        # Max concurrent positions
        active = [t for t in tickets if t.final_weight > 1e-6]
        if len(active) > 8:
            active.sort(key=lambda t: t.conviction_score, reverse=True)
            for t in active[8:]:
                t.final_weight = 0.0
                for leg in t.legs:
                    leg.notional_weight = 0.0

        # Gross exposure check
        gross = sum(t.final_weight for t in tickets)
        max_allowed = self.max_gross * regime_size_cap

        if gross > max_allowed and gross > 0:
            scale = max_allowed / gross
            for t in tickets:
                t.final_weight = round(t.final_weight * scale, 6)
                for leg in t.legs:
                    leg.notional_weight = round(leg.notional_weight * scale, 6)

        # Net exposure constraint: max 30% net
        long_w = sum(t.final_weight for t in tickets if t.direction == "LONG" and t.final_weight > 0)
        short_w = sum(t.final_weight for t in tickets if t.direction == "SHORT" and t.final_weight > 0)
        net = abs(long_w - short_w)
        if net > 0.30:
            # Scale down the larger side
            excess_dir = "LONG" if long_w > short_w else "SHORT"
            excess_tickets = sorted(
                [t for t in tickets if t.direction == excess_dir and t.final_weight > 0],
                key=lambda t: t.conviction_score,
            )
            while net > 0.20 and excess_tickets:
                t = excess_tickets.pop(0)
                reduction = min(t.final_weight, net - 0.20)
                t.final_weight = round(t.final_weight - reduction, 6)
                for leg in t.legs:
                    leg.notional_weight = round(leg.notional_weight * (t.final_weight / (t.final_weight + reduction)) if t.final_weight + reduction > 0 else 0, 6)
                long_w = sum(t2.final_weight for t2 in tickets if t2.direction == "LONG" and t2.final_weight > 0)
                short_w = sum(t2.final_weight for t2 in tickets if t2.direction == "SHORT" and t2.final_weight > 0)
                net = abs(long_w - short_w)

        # Per-name concentration
        for t in tickets:
            if t.final_weight > self.max_single:
                ratio = self.max_single / t.final_weight
                t.final_weight = round(self.max_single, 6)
                for leg in t.legs:
                    leg.notional_weight = round(leg.notional_weight * ratio, 6)

        return tickets


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(row, key, default=float("nan")):
    """Safely extract a float from a dict-like row."""
    try:
        v = float(row.get(key, default) if hasattr(row, "get") else row[key])
        return v if math.isfinite(v) else default
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Summary for agents / API
# ─────────────────────────────────────────────────────────────────────────────

def trade_book_summary(tickets: List[TradeTicket]) -> dict:
    """Compact serializable summary for Claude / AgentBus."""
    if not tickets:
        return {"trades": [], "n_active": 0, "gross_exposure": 0.0}

    active = [t for t in tickets if t.is_active]
    gross = sum(t.final_weight for t in active)

    # Net Greeks
    net_delta_spy = sum(t.greeks.delta_spy for t in active)
    net_vega = sum(t.greeks.vega_net for t in active)
    net_gamma = sum(t.greeks.gamma_net for t in active)
    net_rho_corr = sum(t.greeks.rho_corr for t in active)
    net_rho_disp = sum(t.greeks.rho_dispersion for t in active)

    return {
        "n_total": len(tickets),
        "n_active": len(active),
        "gross_exposure": round(gross, 4),
        "net_greeks": {
            "delta_spy": round(net_delta_spy, 6),
            "vega": round(net_vega, 6),
            "gamma": round(net_gamma, 6),
            "rho_corr": round(net_rho_corr, 6),
            "rho_dispersion": round(net_rho_disp, 6),
        },
        "trades": [
            {
                "id": t.trade_id,
                "type": t.trade_type,
                "ticker": t.ticker,
                "dir": t.direction,
                "weight": t.final_weight,
                "conviction": t.conviction_score,
                "z": t.entry_z,
                "n_legs": t.n_legs,
                "note": t.pm_note,
            }
            for t in active
        ],
    }
