"""
analytics/paper_trader.py
==========================
Paper Trading Simulator — institutional-grade virtual portfolio management

Manages a virtual portfolio tracking DSS signals + momentum strategies:
  1. Dual signal source: Signal Stack (z-score) + Relative Momentum ranking
  2. Realistic execution: slippage model (3bps + √impact), partial fills
  3. Risk management: portfolio stop-loss, net exposure limits, trailing stops
  4. Profit management: partial profit taking at 50% on 1.5% gain
  5. Daily P&L attribution: SPY beta, sector alpha, slippage drag
  6. Benchmark comparison: portfolio vs SPY buy-and-hold
  7. Full performance analytics: Sharpe, Sortino, Calmar, max DD, recovery
  8. Automated journaling to audit trail + agent bus

State persisted in data/paper_portfolio.json — survives restarts.

Usage:
  python analytics/paper_trader.py --update    # Daily update (prices + signals)
  python analytics/paper_trader.py --status    # Show portfolio status
  python analytics/paper_trader.py --reset     # Reset to $1M
  python analytics/paper_trader.py --backtest  # Backtest momentum on historical data
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics.leverage_engine import LeverageEngine

log = logging.getLogger("paper_trader")

PORTFOLIO_PATH = ROOT / "data" / "paper_portfolio.json"
INITIAL_CAPITAL = 1_000_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PaperPosition:
    """Open position in the virtual portfolio."""
    trade_id: str
    ticker: str
    direction: str           # LONG / SHORT
    entry_date: str
    entry_price: float
    notional: float
    weight: float
    entry_z: float
    conviction: float
    regime_at_entry: str
    signal_source: str = "dss"  # "dss" or "momentum"

    # Execution details
    slippage_bps: float = 0.0
    fill_price: float = 0.0  # Actual fill (after slippage)

    # Updated daily
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    days_held: int = 0
    current_z: float = 0.0
    high_water_mark: float = 0.0  # Best P&L % seen (for trailing stop)
    partial_taken: bool = False    # Whether 50% partial profit was taken


@dataclass
class PaperTrade:
    """Closed trade with full attribution."""
    trade_id: str
    ticker: str
    direction: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    notional: float
    realized_pnl: float
    realized_pnl_pct: float
    holding_days: int
    exit_reason: str
    conviction: float
    signal_source: str = "dss"
    slippage_cost: float = 0.0  # Total slippage (entry + exit) in $
    # Attribution
    spy_contribution: float = 0.0  # P&L from SPY beta exposure
    alpha_contribution: float = 0.0  # P&L from sector alpha


@dataclass
class DailySnapshot:
    """Daily portfolio snapshot for performance tracking."""
    date: str
    total_value: float
    pnl_pct: float
    n_positions: int
    cash: float
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    spy_return: float = 0.0       # SPY return on this day
    portfolio_return: float = 0.0  # Portfolio return on this day
    alpha_return: float = 0.0      # portfolio_return - spy_return * beta
    leverage: float = 1.0
    n_entries: int = 0
    n_exits: int = 0


@dataclass
class PaperPortfolio:
    """Full portfolio state."""
    capital: float = INITIAL_CAPITAL
    cash: float = INITIAL_CAPITAL
    positions: List[dict] = field(default_factory=list)
    closed_trades: List[dict] = field(default_factory=list)
    daily_snapshots: List[dict] = field(default_factory=list)
    created_date: str = ""
    last_update: str = ""

    # Performance
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0

    # Benchmark
    spy_total_return: float = 0.0
    alpha_vs_spy: float = 0.0    # Portfolio return - SPY return

    # Attribution totals
    total_slippage_cost: float = 0.0
    total_spy_pnl: float = 0.0
    total_alpha_pnl: float = 0.0

    # Momentum strategy tracking
    momentum_entries: int = 0
    dss_entries: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Execution Model
# ─────────────────────────────────────────────────────────────────────────────

def compute_slippage(notional: float, direction: str, price: float,
                     daily_volume: float = 10_000_000) -> Tuple[float, float]:
    """
    Realistic slippage model for sector ETFs.

    Components:
      1. Spread cost: 1-2 bps (sector ETFs are liquid)
      2. Market impact: Almgren-Chriss √(notional/ADV) × 5bps
      3. Timing cost: 1bps (uncertainty of execution)

    Returns (fill_price, slippage_bps).
    """
    spread_bps = 1.5
    participation = notional / max(daily_volume, 1_000_000)
    impact_bps = 5.0 * math.sqrt(participation)
    timing_bps = 1.0

    total_bps = spread_bps + impact_bps + timing_bps
    slip_pct = total_bps / 10_000

    sign = 1 if direction == "LONG" else -1
    fill_price = price * (1 + sign * slip_pct)

    return fill_price, total_bps


# ─────────────────────────────────────────────────────────────────────────────
# Paper Trader Engine
# ─────────────────────────────────────────────────────────────────────────────

class PaperTrader:
    """
    Institutional-grade paper trading simulator.

    Dual signal sources:
      1. DSS Signal Stack (z-score based, from TradeTickets)
      2. Relative Momentum (cross-sectional sector ranking)

    Usage:
        trader = PaperTrader(settings)
        trader.load()
        trader.daily_update(prices, signal_results, trade_tickets, regime_safety)
        trader.save()
    """

    # Configuration
    MAX_POSITIONS = 10
    MAX_SINGLE_WEIGHT = 0.20
    PORTFOLIO_STOP_PCT = -0.05    # 5% portfolio stop-loss
    NET_EXPOSURE_LIMIT = 0.30     # 30% max net exposure
    TRAILING_STOP_PCT = 0.015     # 1.5% trailing stop from HWM
    PROFIT_TARGET_PCT = 0.015     # 1.5% → take 50% partial profit
    HARD_STOP_PCT = -0.03         # 3% hard stop loss
    TIME_EXIT_DAYS = 25           # Max holding period
    MOMENTUM_LOOKBACK = 21        # Days for momentum ranking
    MOMENTUM_TOP_N = 3            # Top/bottom sectors for momentum

    def __init__(self, settings=None):
        self.portfolio = PaperPortfolio()
        self.settings = settings
        self._leverage_engine = LeverageEngine(
            base_capital=INITIAL_CAPITAL,
            max_leverage=getattr(settings, 'max_leverage', 5.0) if settings else 5.0,
        )
        self._current_leverage = 1.0

        # Override from settings if available
        if settings:
            self.MOMENTUM_LOOKBACK = getattr(settings, 'momentum_lookback', 21)
            self.MOMENTUM_TOP_N = getattr(settings, 'momentum_top_n', 3)
            self.TIME_EXIT_DAYS = getattr(settings, 'signal_optimal_hold', 25)

    # ── Persistence ──────────────────────────────────────────────

    def load(self) -> None:
        if PORTFOLIO_PATH.exists():
            try:
                data = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
                self.portfolio = PaperPortfolio(**data)
                log.info("Loaded paper portfolio: %d positions, %d closed trades",
                         len(self.portfolio.positions), len(self.portfolio.closed_trades))
            except Exception as e:
                log.warning("Failed to load portfolio: %s — starting fresh", e)
                self.portfolio = PaperPortfolio(created_date=date.today().isoformat())
        else:
            self.portfolio = PaperPortfolio(created_date=date.today().isoformat())

    def save(self) -> None:
        self.portfolio.last_update = datetime.now(timezone.utc).isoformat()
        PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
        PORTFOLIO_PATH.write_text(
            json.dumps(asdict(self.portfolio), indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Saved paper portfolio to %s", PORTFOLIO_PATH.name)

    def reset(self) -> None:
        self.portfolio = PaperPortfolio(
            created_date=date.today().isoformat(),
            capital=INITIAL_CAPITAL,
            cash=INITIAL_CAPITAL,
        )
        self.save()
        log.info("Portfolio reset to $%s", f"{INITIAL_CAPITAL:,.0f}")

    # ── Daily Update (main entry point) ─────────────────────────

    def daily_update(
        self,
        prices: pd.DataFrame,
        signal_results: Optional[list] = None,
        trade_tickets: Optional[list] = None,
        regime_safety=None,
    ) -> dict:
        """
        Full daily update cycle:
          1. Mark-to-market all positions
          2. Risk checks: portfolio stop, net exposure
          3. Exit logic: trailing stop, profit target, time exit, hard stop, regime kill
          4. Partial profit taking
          5. New entries: DSS signals + momentum ranking
          6. Compute daily snapshot with attribution
          7. Update performance metrics
        """
        today = date.today().isoformat()
        actions = {"entries": [], "exits": [], "partials": [], "updates": []}

        safety_score = regime_safety.regime_safety_score if regime_safety else 1.0
        safety_label = regime_safety.label if regime_safety else "SAFE"

        # ── 1. Mark-to-market ────────────────────────────────────────
        for pos in self.portfolio.positions:
            ticker = pos["ticker"]
            if ticker in prices.columns:
                current = float(prices[ticker].dropna().iloc[-1])
                entry = pos.get("fill_price") or pos["entry_price"]
                sign = 1.0 if pos["direction"] == "LONG" else -1.0

                pos["current_price"] = current
                pos["unrealized_pnl"] = sign * (current - entry) / entry * pos["notional"]
                pos["unrealized_pnl_pct"] = sign * (current - entry) / entry
                pos["days_held"] = (date.today() - date.fromisoformat(pos["entry_date"])).days

                # Update high-water mark for trailing stop
                if pos["unrealized_pnl_pct"] > pos.get("high_water_mark", 0):
                    pos["high_water_mark"] = pos["unrealized_pnl_pct"]

                actions["updates"].append(pos["trade_id"])

        # ── 2. Portfolio-level risk checks ───────────────────────────
        to_close: List[Tuple[dict, str]] = []

        total_unrealized = sum(p.get("unrealized_pnl", 0) for p in self.portfolio.positions)
        portfolio_dd = total_unrealized / self.portfolio.capital if self.portfolio.capital > 0 else 0

        if portfolio_dd <= self.PORTFOLIO_STOP_PCT:
            log.warning("PORTFOLIO STOP-LOSS: DD=%.1f%% — closing ALL", portfolio_dd * 100)
            for pos in self.portfolio.positions:
                to_close.append((pos, "PORTFOLIO_STOP_LOSS"))

        # Net exposure constraint
        if not to_close:
            long_n = sum(p["notional"] for p in self.portfolio.positions if p["direction"] == "LONG")
            short_n = sum(p["notional"] for p in self.portfolio.positions if p["direction"] == "SHORT")
            net_exp = (long_n - short_n) / self.portfolio.capital if self.portfolio.capital > 0 else 0

            if abs(net_exp) > self.NET_EXPOSURE_LIMIT:
                excess_dir = "LONG" if net_exp > 0 else "SHORT"
                excess = sorted(
                    [p for p in self.portfolio.positions if p["direction"] == excess_dir],
                    key=lambda p: abs(p.get("unrealized_pnl_pct", 0)),
                )
                while abs(net_exp) > 0.20 and excess:
                    pos = excess.pop(0)
                    to_close.append((pos, "NET_EXPOSURE_LIMIT"))
                    if pos["direction"] == "LONG":
                        long_n -= pos["notional"]
                    else:
                        short_n -= pos["notional"]
                    net_exp = (long_n - short_n) / self.portfolio.capital

        # ── 3. Position-level exits ──────────────────────────────────
        if not to_close:
            for pos in self.portfolio.positions:
                reason = self._check_exit(pos, safety_label, safety_score)
                if reason:
                    to_close.append((pos, reason))

        # ── 4. Partial profit taking ─────────────────────────────────
        for pos in self.portfolio.positions:
            if pos.get("partial_taken"):
                continue
            if pos["unrealized_pnl_pct"] >= self.PROFIT_TARGET_PCT:
                # Take 50% off the table
                half_notional = pos["notional"] * 0.5
                half_pnl = pos["unrealized_pnl"] * 0.5
                pos["notional"] -= half_notional
                pos["unrealized_pnl"] -= half_pnl
                pos["partial_taken"] = True
                self.portfolio.cash += half_notional + half_pnl
                actions["partials"].append({
                    "trade_id": pos["trade_id"],
                    "amount": half_notional,
                    "pnl": half_pnl,
                })
                log.info("PARTIAL: 50%% of %s %s — locked $%.0f profit",
                         pos["direction"], pos["ticker"], half_pnl)

        # ── Execute exits ────────────────────────────────────────────
        for pos, reason in to_close:
            if pos not in self.portfolio.positions:
                continue
            self._close_position(pos, reason, prices, actions, today)

        # ── 5. Compute leverage target ───────────────────────────────
        vix_level = self._get_vix(prices)
        leverage_result = self._leverage_engine.compute_target_leverage(
            regime=safety_label or "NORMAL",
            vix=vix_level if math.isfinite(vix_level) else 20.0,
            current_dd_pct=abs(self.portfolio.max_drawdown),
            strategy_sharpe=max(self.portfolio.sharpe, 0.5),
        )
        self._current_leverage = max(0.1, leverage_result.target_leverage)

        # ── 6. New entries: DSS signals ──────────────────────────────
        if trade_tickets:
            self._enter_dss_signals(trade_tickets, prices, safety_label, today, actions)

        # ── 7. New entries: Momentum ranking ─────────────────────────
        self._enter_momentum_signals(prices, safety_label, today, actions)

        # ── 8. Daily snapshot + attribution ──────────────────────────
        self._record_snapshot(prices, today, actions)

        # ── 9. Performance metrics ───────────────────────────────────
        self._update_performance(prices)

        # ── 10. Auto-journal ─────────────────────────────────────────
        self._auto_journal(actions, today)

        return actions

    def _check_exit(self, pos: dict, safety_label: str, safety_score: float) -> Optional[str]:
        """Check all exit conditions for a position."""
        # Time exit
        if pos["days_held"] >= self.TIME_EXIT_DAYS:
            return "TIME_EXIT"

        # Regime kill
        if safety_label in ("KILLED", "DANGER") or safety_score < 0.1:
            return "REGIME_EXIT"

        # Hard stop loss
        if pos["unrealized_pnl_pct"] <= self.HARD_STOP_PCT:
            return "STOP_LOSS"

        # Trailing stop (only if position was in profit)
        hwm = pos.get("high_water_mark", 0)
        if hwm > 0.005:  # Was at least 0.5% in profit
            drawdown_from_hwm = hwm - pos["unrealized_pnl_pct"]
            if drawdown_from_hwm >= self.TRAILING_STOP_PCT:
                return "TRAILING_STOP"

        return None

    def _close_position(self, pos: dict, reason: str, prices: pd.DataFrame,
                        actions: dict, today: str) -> None:
        """Close a position with exit slippage."""
        ticker = pos["ticker"]
        current_price = pos["current_price"]

        # Exit slippage (reverse direction)
        exit_dir = "SHORT" if pos["direction"] == "LONG" else "LONG"
        fill_price, exit_slip_bps = compute_slippage(
            pos["notional"], exit_dir, current_price,
        )

        # Recompute final P&L with exit slippage
        entry = pos.get("fill_price") or pos["entry_price"]
        sign = 1.0 if pos["direction"] == "LONG" else -1.0
        final_pnl_pct = sign * (fill_price - entry) / entry
        final_pnl = final_pnl_pct * pos["notional"]

        # P&L attribution: SPY beta vs alpha
        spy_contrib = 0.0
        alpha_contrib = final_pnl
        if "SPY" in prices.columns and pos["days_held"] > 0:
            spy_series = prices["SPY"].dropna()
            if len(spy_series) >= pos["days_held"]:
                spy_ret = float(spy_series.iloc[-1] / spy_series.iloc[-min(pos["days_held"], len(spy_series))] - 1)
                spy_contrib = sign * spy_ret * pos["notional"]
                alpha_contrib = final_pnl - spy_contrib

        total_slip_cost = (pos.get("slippage_bps", 0) + exit_slip_bps) / 10_000 * pos["notional"]

        trade = {
            "trade_id": pos["trade_id"],
            "ticker": ticker,
            "direction": pos["direction"],
            "entry_date": pos["entry_date"],
            "exit_date": today,
            "entry_price": entry,
            "exit_price": fill_price,
            "notional": pos["notional"],
            "realized_pnl": round(final_pnl, 2),
            "realized_pnl_pct": round(final_pnl_pct, 6),
            "holding_days": pos["days_held"],
            "exit_reason": reason,
            "conviction": pos.get("conviction", 0),
            "signal_source": pos.get("signal_source", "dss"),
            "slippage_cost": round(total_slip_cost, 2),
            "spy_contribution": round(spy_contrib, 2),
            "alpha_contribution": round(alpha_contrib, 2),
        }
        self.portfolio.closed_trades.append(trade)
        self.portfolio.cash += pos["notional"] + final_pnl
        self.portfolio.total_slippage_cost += total_slip_cost
        self.portfolio.total_spy_pnl += spy_contrib
        self.portfolio.total_alpha_pnl += alpha_contrib

        if pos in self.portfolio.positions:
            self.portfolio.positions.remove(pos)

        actions["exits"].append(trade)
        log.info("CLOSED: %s %s | P&L: $%.0f (%.1f%%) | Alpha: $%.0f | Reason: %s",
                 pos["direction"], ticker, final_pnl, final_pnl_pct * 100,
                 alpha_contrib, reason)

    def _enter_dss_signals(self, trade_tickets, prices, safety_label, today, actions):
        """Enter positions from DSS trade tickets."""
        existing = {p["ticker"] for p in self.portfolio.positions}
        for ticket in trade_tickets:
            if not ticket.is_active or ticket.ticker in existing:
                continue
            if len(self.portfolio.positions) >= self.MAX_POSITIONS:
                break

            ticker = ticket.ticker
            if ticker not in prices.columns:
                continue

            current_price = float(prices[ticker].dropna().iloc[-1])
            capped_weight = min(abs(ticket.final_weight), self.MAX_SINGLE_WEIGHT)
            notional = self.portfolio.capital * capped_weight * self._current_leverage

            if notional > self.portfolio.cash:
                continue

            fill_price, slip_bps = compute_slippage(notional, ticket.direction, current_price)

            pos = {
                "trade_id": f"PT_{ticker}_{ticket.direction}_{today}",
                "ticker": ticker,
                "direction": ticket.direction,
                "entry_date": today,
                "entry_price": current_price,
                "fill_price": fill_price,
                "slippage_bps": round(slip_bps, 1),
                "notional": notional,
                "weight": ticket.final_weight,
                "entry_z": ticket.entry_z,
                "conviction": ticket.conviction_score,
                "regime_at_entry": safety_label,
                "signal_source": "dss",
                "current_price": current_price,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "days_held": 0,
                "current_z": ticket.entry_z,
                "high_water_mark": 0.0,
                "partial_taken": False,
            }
            self.portfolio.positions.append(pos)
            self.portfolio.cash -= notional
            self.portfolio.dss_entries += 1
            existing.add(ticker)
            actions["entries"].append(pos["trade_id"])
            log.info("OPENED [DSS]: %s %s @ $%.2f (fill $%.2f, slip %.1fbps) | $%.0f",
                     ticket.direction, ticker, current_price, fill_price, slip_bps, notional)

    def _enter_momentum_signals(self, prices, safety_label, today, actions):
        """Enter positions from cross-sectional momentum ranking."""
        if safety_label in ("KILLED", "DANGER"):
            return

        existing = {p["ticker"] for p in self.portfolio.positions}
        sectors = []
        if self.settings:
            sectors = [s for s in self.settings.sector_list() if s in prices.columns]
        if not sectors or len(sectors) < 5:
            return

        # Compute momentum ranking
        lookback = self.MOMENTUM_LOOKBACK
        if len(prices) < lookback + 5:
            return

        log_rets = np.log(prices[sectors] / prices[sectors].shift(1)).dropna()
        if len(log_rets) < lookback:
            return

        spy_col = "SPY" if "SPY" in prices.columns else None
        if spy_col:
            spy_ret = log_rets.iloc[-lookback:].get(spy_col, pd.Series(0, index=log_rets.index[-lookback:]))
        else:
            spy_ret = 0

        # Relative momentum = sector return - SPY return over lookback
        mom = {}
        for s in sectors:
            if s in log_rets.columns:
                sec_ret = float(log_rets[s].iloc[-lookback:].sum())
                spy_r = float(spy_ret.sum()) if isinstance(spy_ret, pd.Series) else 0
                mom[s] = sec_ret - spy_r

        if not mom:
            return

        # Rank: highest momentum = rank 1
        ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
        top_n = self.MOMENTUM_TOP_N

        for i, (ticker, mom_val) in enumerate(ranked):
            if i < top_n:
                direction = "LONG"
            elif i >= len(ranked) - top_n:
                direction = "SHORT"
            else:
                continue

            if ticker in existing or len(self.portfolio.positions) >= self.MAX_POSITIONS:
                continue

            current_price = float(prices[ticker].dropna().iloc[-1])

            # Vol-scaled sizing
            vol = float(log_rets[ticker].iloc[-60:].std() * np.sqrt(252)) if len(log_rets) >= 60 else 0.15
            weight = min(self.MAX_SINGLE_WEIGHT, 0.10 * 0.15 / max(vol, 0.05))
            notional = self.portfolio.capital * weight * self._current_leverage

            if notional > self.portfolio.cash:
                continue

            fill_price, slip_bps = compute_slippage(notional, direction, current_price)

            pos = {
                "trade_id": f"MOM_{ticker}_{direction}_{today}",
                "ticker": ticker,
                "direction": direction,
                "entry_date": today,
                "entry_price": current_price,
                "fill_price": fill_price,
                "slippage_bps": round(slip_bps, 1),
                "notional": notional,
                "weight": weight,
                "entry_z": 0.0,
                "conviction": abs(mom_val) * 10,  # Normalize momentum to conviction-like score
                "regime_at_entry": safety_label,
                "signal_source": "momentum",
                "current_price": current_price,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "days_held": 0,
                "current_z": 0.0,
                "high_water_mark": 0.0,
                "partial_taken": False,
            }
            self.portfolio.positions.append(pos)
            self.portfolio.cash -= notional
            self.portfolio.momentum_entries += 1
            existing.add(ticker)
            actions["entries"].append(pos["trade_id"])
            log.info("OPENED [MOM]: %s %s @ $%.2f (mom=%.3f, rank=%d) | $%.0f",
                     direction, ticker, current_price, mom_val, i + 1, notional)

    def _record_snapshot(self, prices, today, actions):
        """Record daily snapshot with exposure and attribution."""
        total_value = self.portfolio.cash + sum(
            p["notional"] + p["unrealized_pnl"] for p in self.portfolio.positions
        )
        self.portfolio.total_pnl = total_value - self.portfolio.capital
        self.portfolio.total_pnl_pct = self.portfolio.total_pnl / self.portfolio.capital

        long_exp = sum(p["notional"] for p in self.portfolio.positions if p["direction"] == "LONG")
        short_exp = sum(p["notional"] for p in self.portfolio.positions if p["direction"] == "SHORT")
        cap = self.portfolio.capital

        # SPY return today
        spy_ret = 0.0
        if "SPY" in prices.columns:
            spy = prices["SPY"].dropna()
            if len(spy) >= 2:
                spy_ret = float(spy.iloc[-1] / spy.iloc[-2] - 1)

        # Portfolio return today
        prev = self.portfolio.daily_snapshots[-1]["total_value"] if self.portfolio.daily_snapshots else self.portfolio.capital
        port_ret = (total_value - prev) / prev if prev > 0 else 0

        snapshot = {
            "date": today,
            "total_value": round(total_value, 2),
            "pnl_pct": round(self.portfolio.total_pnl_pct, 6),
            "n_positions": len(self.portfolio.positions),
            "cash": round(self.portfolio.cash, 2),
            "long_exposure": round(long_exp / cap, 4) if cap > 0 else 0,
            "short_exposure": round(short_exp / cap, 4) if cap > 0 else 0,
            "net_exposure": round((long_exp - short_exp) / cap, 4) if cap > 0 else 0,
            "gross_exposure": round((long_exp + short_exp) / cap, 4) if cap > 0 else 0,
            "spy_return": round(spy_ret, 6),
            "portfolio_return": round(port_ret, 6),
            "alpha_return": round(port_ret - spy_ret, 6),
            "leverage": round(self._current_leverage, 2),
            "n_entries": len(actions.get("entries", [])),
            "n_exits": len(actions.get("exits", [])),
        }
        self.portfolio.daily_snapshots.append(snapshot)

        # Keep max 750 snapshots (3 years)
        if len(self.portfolio.daily_snapshots) > 750:
            self.portfolio.daily_snapshots = self.portfolio.daily_snapshots[-750:]

    def _update_performance(self, prices):
        """Compute performance metrics from snapshots."""
        closed = self.portfolio.closed_trades
        self.portfolio.n_trades = len(closed)
        self.portfolio.n_wins = sum(1 for t in closed if t.get("realized_pnl", 0) > 0)
        self.portfolio.win_rate = self.portfolio.n_wins / max(1, self.portfolio.n_trades)

        # Drawdown from snapshots
        values = [s["total_value"] for s in self.portfolio.daily_snapshots]
        if values:
            peak = values[0]
            worst_dd = 0.0
            for v in values:
                peak = max(peak, v)
                dd = (v - peak) / peak if peak > 0 else 0
                worst_dd = min(worst_dd, dd)
            self.portfolio.max_drawdown = worst_dd

        # Sharpe, Sortino, Calmar from daily returns
        snaps = self.portfolio.daily_snapshots
        if len(snaps) >= 20:
            daily_rets = np.array([s.get("portfolio_return", 0) for s in snaps[-252:]])
            daily_rets = daily_rets[np.isfinite(daily_rets)]
            if len(daily_rets) >= 10:
                mu = float(daily_rets.mean())
                sigma = float(daily_rets.std())
                self.portfolio.sharpe = mu / sigma * np.sqrt(252) if sigma > 1e-10 else 0

                # Sortino (downside deviation)
                downside = daily_rets[daily_rets < 0]
                dd_sigma = float(downside.std()) if len(downside) > 5 else sigma
                self.portfolio.sortino = mu / dd_sigma * np.sqrt(252) if dd_sigma > 1e-10 else 0

                # Calmar
                ann_ret = mu * 252
                self.portfolio.calmar = ann_ret / abs(self.portfolio.max_drawdown) if abs(self.portfolio.max_drawdown) > 1e-6 else 0

        # Benchmark comparison
        if len(snaps) >= 2:
            spy_rets = [s.get("spy_return", 0) for s in snaps]
            self.portfolio.spy_total_return = float(np.sum(spy_rets))
            self.portfolio.alpha_vs_spy = self.portfolio.total_pnl_pct - self.portfolio.spy_total_return

    def _get_vix(self, prices):
        vix_col = next((c for c in prices.columns if "VIX" in c.upper()), None)
        if vix_col and vix_col in prices.columns:
            vix = prices[vix_col].dropna()
            if len(vix) > 0:
                return float(vix.iloc[-1])
        return float("nan")

    def _auto_journal(self, actions: dict, today: str) -> None:
        """Publish trades to audit trail and agent bus."""
        try:
            from scripts.agent_bus import AgentBus
            bus = AgentBus()
            if actions.get("entries") or actions.get("exits") or actions.get("partials"):
                bus.publish("paper_trader", {
                    "date": today,
                    "entries": len(actions.get("entries", [])),
                    "exits": len(actions.get("exits", [])),
                    "partials": len(actions.get("partials", [])),
                    "n_positions": len(self.portfolio.positions),
                    "total_pnl_pct": round(self.portfolio.total_pnl_pct, 4),
                    "win_rate": round(self.portfolio.win_rate, 3),
                    "sharpe": round(self.portfolio.sharpe, 3),
                    "leverage": round(self._current_leverage, 2),
                    "momentum_entries": self.portfolio.momentum_entries,
                    "dss_entries": self.portfolio.dss_entries,
                })
        except Exception:
            pass

    # ── Status ───────────────────────────────────────────────────

    def status(self) -> str:
        """Comprehensive portfolio status report."""
        p = self.portfolio
        total_val = p.capital + p.total_pnl

        lines = [
            f"{'='*65}",
            f"  PAPER TRADING PORTFOLIO — {date.today().isoformat()}",
            f"{'='*65}",
            f"",
            f"  Capital:      ${p.capital:>12,.0f}",
            f"  Total Value:  ${total_val:>12,.0f}",
            f"  Cash:         ${p.cash:>12,.0f}",
            f"  P&L:          ${p.total_pnl:>12,.0f} ({p.total_pnl_pct:+.2%})",
            f"  Max DD:       {p.max_drawdown:.2%}",
            f"  Sharpe:       {p.sharpe:.2f} | Sortino: {p.sortino:.2f} | Calmar: {p.calmar:.2f}",
            f"  Win Rate:     {p.n_wins}/{p.n_trades} ({p.win_rate:.0%})",
            f"  Leverage:     {self._current_leverage:.2f}x",
            f"",
            f"  ATTRIBUTION:",
            f"    SPY Beta:    ${p.total_spy_pnl:>10,.0f}",
            f"    Alpha:       ${p.total_alpha_pnl:>10,.0f}",
            f"    Slippage:   -${p.total_slippage_cost:>10,.0f}",
            f"    vs SPY B&H:  {p.alpha_vs_spy:+.2%}",
            f"",
            f"  SIGNAL SOURCES:",
            f"    DSS entries:      {p.dss_entries}",
            f"    Momentum entries: {p.momentum_entries}",
            f"",
        ]

        if p.positions:
            lines.append(f"  OPEN POSITIONS ({len(p.positions)}):")
            long_exp = sum(pos["notional"] for pos in p.positions if pos["direction"] == "LONG")
            short_exp = sum(pos["notional"] for pos in p.positions if pos["direction"] == "SHORT")
            lines.append(f"    Long: ${long_exp:,.0f} | Short: ${short_exp:,.0f} | Net: ${long_exp-short_exp:+,.0f}")
            lines.append(f"")
            for pos in sorted(p.positions, key=lambda x: -abs(x.get("unrealized_pnl", 0))):
                pnl_str = f"${pos['unrealized_pnl']:>8,.0f} ({pos['unrealized_pnl_pct']:+.1%})"
                src = "MOM" if pos.get("signal_source") == "momentum" else "DSS"
                partial = " [50%]" if pos.get("partial_taken") else ""
                lines.append(
                    f"    [{src}] {pos['direction']:<5} {pos['ticker']:<5} @ ${pos['entry_price']:.2f} "
                    f"→ ${pos['current_price']:.2f} | {pnl_str} | {pos['days_held']}d{partial}"
                )
        else:
            lines.append("  OPEN POSITIONS: none")

        if p.closed_trades:
            recent = p.closed_trades[-5:]
            lines.append(f"\n  RECENT TRADES (last {len(recent)}):")
            for t in reversed(recent):
                src = "MOM" if t.get("signal_source") == "momentum" else "DSS"
                pnl_str = f"${t['realized_pnl']:>8,.0f} ({t['realized_pnl_pct']:+.1%})"
                lines.append(
                    f"    [{src}] {t['direction']:<5} {t['ticker']:<5} | "
                    f"{t['entry_date']} → {t['exit_date']} | {pnl_str} | {t['exit_reason']}"
                )

        lines.extend(["", f"{'='*65}"])
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    from config.settings import get_settings
    settings = get_settings()
    trader = PaperTrader(settings=settings)
    trader.load()

    if "--reset" in sys.argv:
        trader.reset()
        print("Portfolio reset.")
        return

    if "--status" in sys.argv:
        print(trader.status())
        return

    if "--update" in sys.argv:
        from analytics.stat_arb import QuantEngine
        from analytics.signal_stack import SignalStackEngine
        from analytics.signal_regime_safety import compute_regime_safety_score
        from analytics.trade_structure import TradeStructureEngine, PositionSizingEngine

        engine = QuantEngine(settings)
        engine.load()
        master_df = engine.calculate_conviction_score()
        prices = engine.prices

        safety = compute_regime_safety_score(
            market_state=str(master_df["market_state"].iloc[0]) if "market_state" in master_df.columns else "NORMAL",
            vix_level=float(master_df["vix_level"].iloc[0]) if "vix_level" in master_df.columns else float("nan"),
            credit_z=float(master_df["credit_z"].iloc[0]) if "credit_z" in master_df.columns else float("nan"),
        )

        ss = SignalStackEngine(settings)
        signals = ss.score_from_master_df(
            frob_distortion_z=0.0, market_mode_share=0.3,
            coc_instability_z=0.0, master_df=master_df,
            regime_safety_result=safety,
        )

        ts = TradeStructureEngine(settings)
        tickets = ts.construct_all_trades(signals, master_df=master_df)
        ps = PositionSizingEngine(settings)
        tickets = ps.size_portfolio(tickets, safety.regime_safety_score, safety.size_cap)

        actions = trader.daily_update(prices, signals, tickets, safety)
        trader.save()

        print(f"\nActions: {len(actions['entries'])} entries, {len(actions['exits'])} exits, "
              f"{len(actions.get('partials', []))} partials, {len(actions['updates'])} updates")
        print(trader.status())
        return

    print("Usage: python analytics/paper_trader.py [--update|--status|--reset]")


if __name__ == "__main__":
    main()
