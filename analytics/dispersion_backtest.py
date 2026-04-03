"""
analytics/dispersion_backtest.py
================================
Simulates actual Dispersion Trade P&L — the core missing piece.

A dispersion trade = SHORT index variance + LONG single-name variance.
Profits when realized correlation < implied correlation.

This module backtests actual trade P&L, not just signal quality:
  1. Entry: when signal stack says "enter" (S^dist × S^disloc × S^mr × S^safe > θ)
  2. Hold: carry P&L = Σ(vega_i × ΔIV_i) + theta decay + gamma P&L
  3. Exit: when residual compresses, time expires, or regime kills

Three trade types:
  A. Variance swap replication: short SPY var, long sector var
  B. Straddle dispersion: short SPY straddle, long sector straddles
  C. Simplified: long/short sector relative value (what we can actually trade with ETFs)

Ref: Bossu (2006) — "An Introduction to Variance Swaps"
Ref: Jacquier & Slaoui (2007) — "Variance Dispersion and Correlation Swaps"
Ref: Derman et al. (1999) — "More Than You Ever Wanted to Know About Volatility Swaps"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DispersionTrade:
    """A single dispersion trade from entry to exit."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    trade_type: str           # "variance_dispersion" / "rv_sector" / "straddle"
    long_leg: str             # e.g., "XLK" (sector we're long vol on)
    short_leg: str            # e.g., "SPY" (index we're short vol on)
    entry_iv_long: float      # IV of long leg at entry
    entry_iv_short: float     # IV of short leg at entry
    entry_rv_long: float      # RV of long leg at entry
    entry_rv_short: float     # RV of short leg at entry
    exit_iv_long: float
    exit_iv_short: float
    exit_rv_long: float
    exit_rv_short: float
    holding_days: int
    vega_pnl: float           # P&L from IV changes
    theta_pnl: float          # P&L from time decay
    gamma_pnl: float          # P&L from realized moves
    total_pnl: float          # Net P&L (% of notional)
    exit_reason: str          # "signal_compress" / "time_exit" / "regime_kill" / "stop_loss"
    regime_at_entry: str
    regime_at_exit: str
    conviction_at_entry: float


@dataclass
class DispersionBacktestResult:
    """Full backtest result with trade-level and portfolio-level metrics."""
    trades: List[DispersionTrade]
    # Portfolio metrics
    total_pnl: float
    sharpe: float
    win_rate: float
    avg_pnl_per_trade: float
    max_drawdown: float
    calmar: float
    total_trades: int
    avg_holding_days: float
    # Breakdown
    pnl_by_regime: Dict[str, float]
    pnl_by_sector: Dict[str, float]
    pnl_by_exit_reason: Dict[str, float]
    # Time series
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    # Component P&L
    total_vega_pnl: float
    total_theta_pnl: float
    total_gamma_pnl: float


# ─────────────────────────────────────────────────────────────────────────────
# Core: Variance swap P&L approximation
# ─────────────────────────────────────────────────────────────────────────────

def _ewma_vol(returns: pd.Series, span: int = 20) -> pd.Series:
    """EWMA volatility (annualized)."""
    return returns.ewm(span=span).std() * np.sqrt(252)


def _realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling realized volatility (annualized)."""
    return returns.rolling(window).std() * np.sqrt(252)


def _implied_vol_proxy(vix: pd.Series, beta: float, rv_sector: pd.Series, rv_spy: pd.Series) -> pd.Series:
    """
    Proxy sector implied vol from VIX:
      IV_sector ≈ β × VIX + (RV_sector - β × RV_SPY) × shrinkage

    This captures:
    - Systematic component: β × VIX
    - Idiosyncratic premium: sector's excess vol over beta-expected
    """
    iv_vix = vix / 100  # VIX is in percentage points
    systematic = beta * iv_vix
    idio = rv_sector - beta * rv_spy
    shrinkage = 0.5  # blend realized idio with zero (conservative)
    return systematic + shrinkage * idio


def _variance_swap_pnl(
    iv_entry: float,
    rv_realized: float,
    T: float = 21 / 252,
    notional: float = 1.0,
) -> float:
    """
    P&L of a variance swap:
      P&L = notional × T × (σ²_realized - K²_var)
    where K_var = IV at entry.

    Long variance profits when RV > IV.
    Short variance profits when RV < IV.
    """
    if not (math.isfinite(iv_entry) and math.isfinite(rv_realized)):
        return 0.0
    return notional * T * (rv_realized ** 2 - iv_entry ** 2)


def _dispersion_pnl(
    iv_index_entry: float,
    iv_sector_entry: float,
    rv_index: float,
    rv_sector: float,
    implied_corr_entry: float,
    realized_corr: float,
    T: float = 21 / 252,
    notional: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Dispersion trade P&L decomposition:
      Total = Vega P&L + Theta P&L + Gamma P&L

    Vega P&L: from IV changes (mean reversion of IV spread)
    Theta P&L: from time decay (short index theta > long sector theta if IV_index > IV_sector)
    Gamma P&L: from correlation dropping (realized moves are less correlated than implied)

    The key insight: dispersion profits when realized_corr < implied_corr.
    P&L ≈ notional × (implied_corr - realized_corr) × σ_avg² × T
    """
    if not all(math.isfinite(x) for x in [iv_index_entry, iv_sector_entry, rv_index, rv_sector,
                                            implied_corr_entry, realized_corr]):
        return 0.0, 0.0, 0.0, 0.0

    # Vega P&L: short index vol + long sector vol
    # If sector IV drops less than index IV, we profit
    vega = notional * T * (rv_sector ** 2 - iv_sector_entry ** 2) - \
           notional * T * (rv_index ** 2 - iv_index_entry ** 2)

    # Theta P&L: net time decay (short index decays faster if higher IV)
    theta = notional * T * 0.5 * (iv_index_entry ** 2 - iv_sector_entry ** 2) * 0.1  # simplified

    # Gamma P&L: correlation breakdown benefit
    sigma_avg = 0.5 * (iv_index_entry + iv_sector_entry)
    gamma = notional * T * (implied_corr_entry - realized_corr) * sigma_avg ** 2

    total = vega + theta + gamma
    return total, vega, theta, gamma


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────

class DispersionBacktester:
    """
    Walk-forward backtest of actual dispersion trade P&L.

    Parameters
    ----------
    prices : pd.DataFrame — full price panel (sectors + SPY + VIX)
    sectors : list — sector ETF tickers
    lookback : int — window for realized vol (default 20d)
    hold_period : int — max holding period (default 21d = 1 month)
    z_entry : float — min z-score to enter (default 0.8)
    z_exit : float — z-score to exit (default 0.3)
    max_positions : int — max simultaneous positions (default 3)
    regime_kill_vix : float — VIX level that kills all trades (default 35)
    stop_loss_pct : float — max loss per trade (default 0.03 = 3%)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        sectors: Optional[List[str]] = None,
        lookback: int = 20,
        hold_period: int = 21,
        z_entry: float = 0.8,
        z_exit: float = 0.3,
        max_positions: int = 3,
        regime_kill_vix: float = 35.0,
        stop_loss_pct: float = 0.03,
    ):
        self.prices = prices
        self.sectors = sectors or ['XLC', 'XLY', 'XLP', 'XLE', 'XLF',
                                   'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
        self.lookback = lookback
        self.hold_period = hold_period
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.max_positions = max_positions
        self.regime_kill_vix = regime_kill_vix
        self.stop_loss_pct = stop_loss_pct

    def run(self) -> DispersionBacktestResult:
        """Run the full dispersion backtest."""
        prices = self.prices
        spy = prices["SPY"]
        vix = prices["^VIX"] if "^VIX" in prices.columns else prices.get("VIX", pd.Series(dtype=float))

        # Compute returns and vols
        spy_ret = spy.pct_change()
        spy_rv = _realized_vol(spy_ret, self.lookback)
        spy_iv = vix / 100  # VIX as IV proxy

        sector_rets = {}
        sector_rv = {}
        sector_iv = {}
        sector_betas = {}

        for s in self.sectors:
            if s not in prices.columns:
                continue
            ret = prices[s].pct_change()
            sector_rets[s] = ret
            sector_rv[s] = _realized_vol(ret, self.lookback)

            # Rolling beta to SPY
            cov = ret.rolling(60).cov(spy_ret)
            var = spy_ret.rolling(60).var()
            beta = (cov / var).clip(0.3, 3.0)
            sector_betas[s] = beta

            # IV proxy
            sector_iv[s] = _implied_vol_proxy(vix, beta.iloc[-1] if len(beta) > 0 else 1.0,
                                               sector_rv[s], spy_rv)

        # Rolling correlation (SPY vs each sector)
        sector_corr = {}
        for s in self.sectors:
            if s in sector_rets:
                sector_corr[s] = sector_rets[s].rolling(self.lookback).corr(spy_ret)

        # Average cross-sector correlation
        avg_corr = pd.DataFrame(sector_corr).mean(axis=1)

        # Implied correlation proxy (from VIX and sector vols)
        # ρ_impl = (σ²_I - Σw²σ²) / (Σ_i≠j w_i w_j σ_i σ_j × 2)
        # Simplified: use VRP as proxy for impl_corr - real_corr spread
        vrp = spy_iv ** 2 - spy_rv ** 2  # Variance risk premium

        # PCA residual z-scores for entry signals
        cum_rel = {}
        z_scores = {}
        for s in self.sectors:
            if s in sector_rets:
                cum_rel[s] = np.log(prices[s] / spy).dropna()
                mu = cum_rel[s].rolling(60).mean()
                sigma = cum_rel[s].rolling(60).std()
                z_scores[s] = ((cum_rel[s] - mu) / sigma).replace([np.inf, -np.inf], 0)

        # Walk through time
        trades: List[DispersionTrade] = []
        daily_pnl = pd.Series(0.0, index=prices.index)
        open_positions: List[dict] = []
        warmup = max(252, self.lookback * 3)

        for i in range(warmup, len(prices) - self.hold_period):
            date = prices.index[i]
            current_vix = float(vix.iloc[i]) if i < len(vix) else 20.0

            # Determine regime
            if current_vix < 15:
                regime = "CALM"
            elif current_vix < 22:
                regime = "NORMAL"
            elif current_vix < 30:
                regime = "TENSION"
            else:
                regime = "CRISIS"

            # Check exits on open positions
            closed_today = []
            for pos in open_positions:
                days_held = i - pos["entry_idx"]
                s = pos["sector"]

                # Current z-score
                z_now = float(z_scores[s].iloc[i]) if s in z_scores and i < len(z_scores[s]) else 0.0

                # Running P&L
                if s in sector_rv and s in sector_iv:
                    rv_s = float(sector_rv[s].iloc[i]) if i < len(sector_rv[s]) else 0.0
                    rv_spy_now = float(spy_rv.iloc[i]) if i < len(spy_rv) else 0.0
                    corr_now = float(sector_corr.get(s, pd.Series(0.5)).iloc[i]) if s in sector_corr and i < len(sector_corr[s]) else 0.5
                    impl_corr = float(avg_corr.iloc[pos["entry_idx"]]) if pos["entry_idx"] < len(avg_corr) else 0.5

                    total, vega, theta, gamma = _dispersion_pnl(
                        iv_index_entry=pos["entry_iv_spy"],
                        iv_sector_entry=pos["entry_iv_sector"],
                        rv_index=rv_spy_now,
                        rv_sector=rv_s,
                        implied_corr_entry=impl_corr,
                        realized_corr=corr_now,
                        T=days_held / 252,
                    )
                    pos["running_pnl"] = total
                    pos["vega_pnl"] = vega
                    pos["theta_pnl"] = theta
                    pos["gamma_pnl"] = gamma

                # Exit conditions
                exit_reason = None
                if current_vix > self.regime_kill_vix:
                    exit_reason = "regime_kill"
                elif abs(z_now) < self.z_exit:
                    exit_reason = "signal_compress"
                elif days_held >= self.hold_period:
                    exit_reason = "time_exit"
                elif pos["running_pnl"] < -self.stop_loss_pct:
                    exit_reason = "stop_loss"

                if exit_reason:
                    exit_date = prices.index[i]
                    rv_s_exit = float(sector_rv[s].iloc[i]) if s in sector_rv and i < len(sector_rv[s]) else 0.0
                    iv_s_exit = float(sector_iv[s].iloc[i]) if s in sector_iv and i < len(sector_iv[s]) else 0.0
                    rv_spy_exit = float(spy_rv.iloc[i]) if i < len(spy_rv) else 0.0
                    iv_spy_exit = float(spy_iv.iloc[i]) if i < len(spy_iv) else 0.0

                    trade = DispersionTrade(
                        entry_date=prices.index[pos["entry_idx"]],
                        exit_date=exit_date,
                        trade_type="variance_dispersion",
                        long_leg=s,
                        short_leg="SPY",
                        entry_iv_long=pos["entry_iv_sector"],
                        entry_iv_short=pos["entry_iv_spy"],
                        entry_rv_long=pos["entry_rv_sector"],
                        entry_rv_short=pos["entry_rv_spy"],
                        exit_iv_long=iv_s_exit,
                        exit_iv_short=iv_spy_exit,
                        exit_rv_long=rv_s_exit,
                        exit_rv_short=rv_spy_exit,
                        holding_days=days_held,
                        vega_pnl=pos.get("vega_pnl", 0),
                        theta_pnl=pos.get("theta_pnl", 0),
                        gamma_pnl=pos.get("gamma_pnl", 0),
                        total_pnl=pos["running_pnl"],
                        exit_reason=exit_reason,
                        regime_at_entry=pos["regime"],
                        regime_at_exit=regime,
                        conviction_at_entry=pos["conviction"],
                    )
                    trades.append(trade)
                    daily_pnl.iloc[i] += pos["running_pnl"] / max(1, self.max_positions)
                    closed_today.append(pos)

            for p in closed_today:
                open_positions.remove(p)

            # Check entries (only if room for more positions)
            if len(open_positions) < self.max_positions and regime != "CRISIS":
                candidates = []
                for s in self.sectors:
                    if s not in z_scores or any(p["sector"] == s for p in open_positions):
                        continue
                    z = float(z_scores[s].iloc[i]) if i < len(z_scores[s]) else 0.0
                    if not math.isfinite(z) or abs(z) < self.z_entry:
                        continue
                    candidates.append((s, z, abs(z)))

                # Sort by |z| descending — strongest signal first
                candidates.sort(key=lambda x: x[2], reverse=True)

                for s, z, z_abs in candidates[:self.max_positions - len(open_positions)]:
                    iv_s = float(sector_iv[s].iloc[i]) if s in sector_iv and i < len(sector_iv[s]) else 0.0
                    iv_spy_now = float(spy_iv.iloc[i]) if i < len(spy_iv) else 0.0
                    rv_s = float(sector_rv[s].iloc[i]) if s in sector_rv and i < len(sector_rv[s]) else 0.0
                    rv_spy_now = float(spy_rv.iloc[i]) if i < len(spy_rv) else 0.0

                    open_positions.append({
                        "sector": s,
                        "entry_idx": i,
                        "entry_z": z,
                        "entry_iv_sector": iv_s,
                        "entry_iv_spy": iv_spy_now,
                        "entry_rv_sector": rv_s,
                        "entry_rv_spy": rv_spy_now,
                        "regime": regime,
                        "conviction": z_abs / 3.0,  # normalize
                        "running_pnl": 0.0,
                        "vega_pnl": 0.0,
                        "theta_pnl": 0.0,
                        "gamma_pnl": 0.0,
                    })

        # Compute results
        if not trades:
            return DispersionBacktestResult(
                trades=[], total_pnl=0, sharpe=0, win_rate=0,
                avg_pnl_per_trade=0, max_drawdown=0, calmar=0,
                total_trades=0, avg_holding_days=0,
                pnl_by_regime={}, pnl_by_sector={}, pnl_by_exit_reason={},
                equity_curve=pd.Series(dtype=float), drawdown_curve=pd.Series(dtype=float),
                total_vega_pnl=0, total_theta_pnl=0, total_gamma_pnl=0,
            )

        trade_pnls = np.array([t.total_pnl for t in trades])
        total_pnl = float(trade_pnls.sum())
        win_rate = float((trade_pnls > 0).mean())
        avg_pnl = float(trade_pnls.mean())

        # Equity curve
        eq = daily_pnl.cumsum()
        peak = eq.cummax()
        dd = eq - peak
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0

        # Sharpe
        daily_std = daily_pnl[daily_pnl != 0].std()
        daily_mean = daily_pnl[daily_pnl != 0].mean()
        sharpe = float(daily_mean / daily_std * np.sqrt(252)) if daily_std > 1e-10 else 0.0
        calmar = float(sharpe / abs(max_dd)) if abs(max_dd) > 1e-10 else 0.0

        # Breakdowns
        pnl_by_regime = {}
        for t in trades:
            pnl_by_regime[t.regime_at_entry] = pnl_by_regime.get(t.regime_at_entry, 0) + t.total_pnl

        pnl_by_sector = {}
        for t in trades:
            pnl_by_sector[t.long_leg] = pnl_by_sector.get(t.long_leg, 0) + t.total_pnl

        pnl_by_exit = {}
        for t in trades:
            pnl_by_exit[t.exit_reason] = pnl_by_exit.get(t.exit_reason, 0) + t.total_pnl

        return DispersionBacktestResult(
            trades=trades,
            total_pnl=round(total_pnl, 6),
            sharpe=round(sharpe, 4),
            win_rate=round(win_rate, 4),
            avg_pnl_per_trade=round(avg_pnl, 6),
            max_drawdown=round(max_dd, 6),
            calmar=round(calmar, 4),
            total_trades=len(trades),
            avg_holding_days=round(np.mean([t.holding_days for t in trades]), 1),
            pnl_by_regime=pnl_by_regime,
            pnl_by_sector=pnl_by_sector,
            pnl_by_exit_reason=pnl_by_exit,
            equity_curve=eq,
            drawdown_curve=dd,
            total_vega_pnl=round(sum(t.vega_pnl for t in trades), 6),
            total_theta_pnl=round(sum(t.theta_pnl for t in trades), 6),
            total_gamma_pnl=round(sum(t.gamma_pnl for t in trades), 6),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Grid search for optimal parameters
# ─────────────────────────────────────────────────────────────────────────────

def optimize_dispersion_params(
    prices: pd.DataFrame,
    sectors: Optional[List[str]] = None,
) -> Dict:
    """Grid search over dispersion backtest parameters."""
    best_sharpe = -999
    best_params = {}
    results = []

    for lookback in [15, 20, 30]:
        for hold in [15, 21, 30]:
            for z_entry in [0.6, 0.8, 1.0, 1.2]:
                for z_exit in [0.2, 0.3, 0.4]:
                    for max_pos in [2, 3, 4]:
                        bt = DispersionBacktester(
                            prices, sectors,
                            lookback=lookback,
                            hold_period=hold,
                            z_entry=z_entry,
                            z_exit=z_exit,
                            max_positions=max_pos,
                        )
                        r = bt.run()
                        if r.total_trades >= 50:
                            results.append({
                                "lookback": lookback, "hold": hold,
                                "z_entry": z_entry, "z_exit": z_exit,
                                "max_pos": max_pos,
                                "sharpe": r.sharpe, "wr": r.win_rate,
                                "pnl": r.total_pnl, "trades": r.total_trades,
                                "dd": r.max_drawdown,
                            })
                            if r.sharpe > best_sharpe:
                                best_sharpe = r.sharpe
                                best_params = results[-1].copy()

    return {"best": best_params, "all_results": results}


# ═════════════════════════════════════════════════════════════════════════════
# Enhanced Greeks Estimation for Dispersion Trades
# ═════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
import math


@dataclass
class DispersionGreeks:
    """Full Greeks profile for a dispersion trade at a point in time."""
    # Portfolio-level
    net_delta: float = 0.0           # Should be ~0 (delta-neutral by construction)
    net_gamma: float = 0.0           # Long gamma (from sector straddles)
    net_vega: float = 0.0            # Should be ~0 if properly hedged
    net_theta: float = 0.0           # Net time decay (negative = paying)
    # Correlation Greeks
    rho_correlation: float = 0.0     # dP&L/d(correlation) — negative for dispersion
    rho_dispersion: float = 0.0      # dP&L/d(dispersion) — positive
    # Per-component
    index_vega: float = 0.0          # Short index vega (collecting premium)
    sector_vega_total: float = 0.0   # Long sector vega (paying premium)
    vega_mismatch: float = 0.0       # |index_vega| - |sector_vega_total|


def estimate_dispersion_greeks(
    index_price: float,
    sector_prices: Dict[str, float],
    index_iv: float,
    sector_ivs: Dict[str, float],
    sector_weights: Dict[str, float],
    dte: int = 21,
    notional: float = 100_000,
) -> DispersionGreeks:
    """
    Estimate Greeks for a dispersion trade (short index vol + long sector vol).

    Parameters
    ----------
    index_price, sector_prices — current prices
    index_iv, sector_ivs — annualized implied volatilities
    sector_weights — weight of each sector in the dispersion basket
    dte — days to expiration
    notional — dollar notional of the trade
    """
    T = dte / 252
    sqrt_T = math.sqrt(T)

    # Index straddle Greeks (SHORT)
    idx_vega = -notional * index_iv * sqrt_T * 0.01  # Per 1% vol move
    idx_gamma = -notional * 0.4 / (index_price * index_iv * sqrt_T + 0.01)
    idx_theta = notional * index_iv / (2 * sqrt_T * 252 + 0.01) * 0.01

    # Sector straddle Greeks (LONG)
    sec_vega_total = 0.0
    sec_gamma_total = 0.0
    sec_theta_total = 0.0

    for sec, w in sector_weights.items():
        if sec not in sector_ivs or sec not in sector_prices:
            continue
        sec_notional = notional * w
        sec_iv = sector_ivs[sec]
        sec_price = sector_prices[sec]

        sec_vega = sec_notional * sec_iv * sqrt_T * 0.01
        sec_gamma = sec_notional * 0.4 / (sec_price * sec_iv * sqrt_T + 0.01)
        sec_theta = -sec_notional * sec_iv / (2 * sqrt_T * 252 + 0.01) * 0.01

        sec_vega_total += sec_vega
        sec_gamma_total += sec_gamma
        sec_theta_total += sec_theta

    # Correlation sensitivity: dP&L/d(ρ) ≈ -notional × σ_avg × √T
    avg_iv = float(np.mean(list(sector_ivs.values()))) if sector_ivs else 0.15
    rho_corr = -notional * avg_iv * sqrt_T * 0.01

    return DispersionGreeks(
        net_delta=0.0,  # Delta-neutral by construction
        net_gamma=round(sec_gamma_total + idx_gamma, 4),
        net_vega=round(sec_vega_total + idx_vega, 4),
        net_theta=round(sec_theta_total + idx_theta, 4),
        rho_correlation=round(rho_corr, 4),
        rho_dispersion=round(-rho_corr, 4),
        index_vega=round(idx_vega, 4),
        sector_vega_total=round(sec_vega_total, 4),
        vega_mismatch=round(abs(idx_vega) - abs(sec_vega_total), 4),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Skew Impact Model
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SkewImpact:
    """Impact of volatility skew on dispersion trade P&L."""
    skew_cost_bps: float              # Cost from negative skew (buying OTM puts)
    skew_benefit_bps: float           # Benefit from selling OTM index puts
    net_skew_impact_bps: float        # Net P&L impact from skew
    skew_regime: str                  # "FAVORABLE" / "NEUTRAL" / "ADVERSE"


def estimate_skew_impact(
    index_iv: float,
    sector_ivs: Dict[str, float],
    skew_index: float = 100.0,       # CBOE Skew index
    direction: str = "SHORT_INDEX",   # Standard dispersion
) -> SkewImpact:
    """
    Estimate the impact of volatility skew on dispersion trade P&L.

    In a standard dispersion trade (short index vol):
    - Index skew is typically steeper → we BENEFIT from selling rich index puts
    - Sector skew is flatter → lower cost on sector straddles

    When CBOE Skew > 130: index put skew is extremely rich → MORE favorable
    When CBOE Skew < 110: skew is flat → LESS favorable (no premium)
    """
    avg_sector_iv = float(np.mean(list(sector_ivs.values()))) if sector_ivs else 0.15

    # Skew premium estimation
    # Index: richer skew → we collect more premium (BENEFIT)
    skew_premium = max(0, (skew_index - 100) * 0.5)  # bps per point of skew above 100

    # Sector: flatter skew → lower cost
    sector_skew_cost = avg_sector_iv * 2  # Simplified: 2bps per 1% IV

    # Net impact
    net = skew_premium - sector_skew_cost

    if net > 5:
        regime = "FAVORABLE"
    elif net < -5:
        regime = "ADVERSE"
    else:
        regime = "NEUTRAL"

    return SkewImpact(
        skew_cost_bps=round(sector_skew_cost, 1),
        skew_benefit_bps=round(skew_premium, 1),
        net_skew_impact_bps=round(net, 1),
        skew_regime=regime,
    )


from typing import Dict
