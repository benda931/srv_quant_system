"""
analytics/paper_trader.py
==========================
Paper Trading Simulator — מעקב real-time על סיגנלי DSS בלי כסף אמיתי

מנהל פורטפוליו וירטואלי שעוקב אחרי:
  1. כניסות חדשות מ-Signal Stack
  2. יציאות לפי Trade Monitor
  3. P&L יומי מצטבר
  4. היסטוריית טריידים מלאה
  5. ביצועים לעומת benchmark (SPY)

State נשמר ב-data/paper_portfolio.json ונטען בכל הפעלה.

הרצה:
  python analytics/paper_trader.py --update    # עדכון יומי
  python analytics/paper_trader.py --status    # הצגת מצב
  python analytics/paper_trader.py --reset     # איפוס פורטפוליו
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("paper_trader")

PORTFOLIO_PATH = ROOT / "data" / "paper_portfolio.json"
INITIAL_CAPITAL = 1_000_000.0  # $1M virtual capital


# ─────────────────────────────────────────────────────────────────────────────
# Paper Position
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PaperPosition:
    """פוזיציה בפורטפוליו הוירטואלי."""
    trade_id: str
    ticker: str
    direction: str           # LONG / SHORT
    entry_date: str
    entry_price: float
    notional: float          # Dollar amount
    weight: float            # % of portfolio
    entry_z: float
    conviction: float
    regime_at_entry: str

    # Updated daily
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    days_held: int = 0
    current_z: float = 0.0


@dataclass
class PaperTrade:
    """טרייד סגור בפורטפוליו הוירטואלי."""
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


@dataclass
class PaperPortfolio:
    """מצב הפורטפוליו הוירטואלי."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Paper Trader Engine
# ─────────────────────────────────────────────────────────────────────────────

class PaperTrader:
    """
    מנהל פורטפוליו paper trading.

    Usage:
        trader = PaperTrader()
        trader.load()
        trader.daily_update(prices, signal_results, regime_safety)
        trader.save()
        print(trader.status())
    """

    def __init__(self):
        self.portfolio = PaperPortfolio()

    # ── Persistence ──────────────────────────────────────────────

    def load(self) -> None:
        """טעינת state מ-JSON."""
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
        """שמירת state ל-JSON."""
        self.portfolio.last_update = datetime.now(timezone.utc).isoformat()
        PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
        PORTFOLIO_PATH.write_text(
            json.dumps(asdict(self.portfolio), indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Saved paper portfolio to %s", PORTFOLIO_PATH.name)

    def reset(self) -> None:
        """איפוס לפורטפוליו חדש."""
        self.portfolio = PaperPortfolio(
            created_date=date.today().isoformat(),
            capital=INITIAL_CAPITAL,
            cash=INITIAL_CAPITAL,
        )
        self.save()
        log.info("Portfolio reset to $%s", f"{INITIAL_CAPITAL:,.0f}")

    # ── Daily Update ─────────────────────────────────────────────

    def daily_update(
        self,
        prices: pd.DataFrame,
        signal_results: Optional[list] = None,
        trade_tickets: Optional[list] = None,
        regime_safety=None,
    ) -> dict:
        """
        עדכון יומי — מחירים, P&L, אותות כניסה/יציאה חדשים.

        Parameters
        ----------
        prices         : daily close prices
        signal_results : from SignalStackEngine (optional — for new entries)
        trade_tickets  : from TradeStructureEngine (optional)
        regime_safety  : from RegimeSafetyResult (optional)
        """
        today = date.today().isoformat()
        actions = {"entries": [], "exits": [], "updates": []}

        # 1. Update existing positions
        for pos in self.portfolio.positions:
            ticker = pos["ticker"]
            if ticker in prices.columns:
                current = float(prices[ticker].dropna().iloc[-1])
                entry_price = pos["entry_price"]
                sign = 1.0 if pos["direction"] == "LONG" else -1.0

                pos["current_price"] = current
                pos["unrealized_pnl"] = sign * (current - entry_price) / entry_price * pos["notional"]
                pos["unrealized_pnl_pct"] = sign * (current - entry_price) / entry_price
                pos["days_held"] = (date.today() - date.fromisoformat(pos["entry_date"])).days
                actions["updates"].append(pos["trade_id"])

        # 2. Portfolio-level risk checks FIRST
        safety_score = regime_safety.regime_safety_score if regime_safety else 1.0
        safety_label = regime_safety.label if regime_safety else "SAFE"
        to_close = []

        # ── Portfolio drawdown kill-switch (max 5% portfolio loss) ──
        total_unrealized = sum(p.get("unrealized_pnl", 0) for p in self.portfolio.positions)
        portfolio_dd_pct = total_unrealized / self.portfolio.capital if self.portfolio.capital > 0 else 0
        if portfolio_dd_pct <= -0.05:  # 5% portfolio drawdown
            log.warning("🚨 PORTFOLIO STOP-LOSS: DD=%.1f%% — closing ALL positions", portfolio_dd_pct * 100)
            for pos in self.portfolio.positions:
                to_close.append((pos, "PORTFOLIO_STOP_LOSS"))

        # ── Net exposure constraint (max 30% net, target market-neutral) ──
        if not to_close:
            long_notional = sum(p["notional"] for p in self.portfolio.positions if p["direction"] == "LONG")
            short_notional = sum(p["notional"] for p in self.portfolio.positions if p["direction"] == "SHORT")
            net_exposure = (long_notional - short_notional) / self.portfolio.capital if self.portfolio.capital > 0 else 0
            if abs(net_exposure) > 0.30:
                # Close the most directional positions to reduce net
                excess_dir = "LONG" if net_exposure > 0 else "SHORT"
                excess_positions = sorted(
                    [p for p in self.portfolio.positions if p["direction"] == excess_dir],
                    key=lambda p: abs(p.get("unrealized_pnl_pct", 0)),
                )
                while abs(net_exposure) > 0.20 and excess_positions:
                    pos = excess_positions.pop(0)
                    to_close.append((pos, "NET_EXPOSURE_LIMIT"))
                    if pos["direction"] == "LONG":
                        long_notional -= pos["notional"]
                    else:
                        short_notional -= pos["notional"]
                    net_exposure = (long_notional - short_notional) / self.portfolio.capital

        # ── Position-level exit conditions ──
        if not to_close:
            for pos in self.portfolio.positions:
                should_close = False
                reason = ""

                # Time exit (calibrated: 25 days)
                if pos["days_held"] >= 25:
                    should_close, reason = True, "TIME_EXIT"
                # Regime kill
                elif safety_label in ("KILLED", "DANGER") or safety_score < 0.1:
                    should_close, reason = True, "REGIME_EXIT"
                # Profit target: 2% gain
                elif pos["unrealized_pnl_pct"] >= 0.02:
                    should_close, reason = True, "PROFIT_TARGET"
                # Stop loss: 3% loss
                elif pos["unrealized_pnl_pct"] <= -0.03:
                    should_close, reason = True, "STOP_LOSS"

                if should_close:
                    to_close.append((pos, reason))

        for pos, reason in to_close:
            trade = {
                "trade_id": pos["trade_id"],
                "ticker": pos["ticker"],
                "direction": pos["direction"],
                "entry_date": pos["entry_date"],
                "exit_date": today,
                "entry_price": pos["entry_price"],
                "exit_price": pos["current_price"],
                "notional": pos["notional"],
                "realized_pnl": pos["unrealized_pnl"],
                "realized_pnl_pct": pos["unrealized_pnl_pct"],
                "holding_days": pos["days_held"],
                "exit_reason": reason,
                "conviction": pos["conviction"],
            }
            self.portfolio.closed_trades.append(trade)
            self.portfolio.cash += pos["notional"] + pos["unrealized_pnl"]
            self.portfolio.positions.remove(pos)
            actions["exits"].append(trade)
            log.info("CLOSED: %s %s | P&L: $%.0f (%.1f%%) | Reason: %s",
                     pos["direction"], pos["ticker"], pos["unrealized_pnl"],
                     pos["unrealized_pnl_pct"] * 100, reason)

        # 3. Open new positions from signal results
        if trade_tickets:
            existing_tickers = {p["ticker"] for p in self.portfolio.positions}
            for ticket in trade_tickets:
                if not ticket.is_active:
                    continue
                if ticket.ticker in existing_tickers:
                    continue
                if len(self.portfolio.positions) >= 8:  # Max 8 positions
                    break

                ticker = ticket.ticker
                if ticker not in prices.columns:
                    continue

                current_price = float(prices[ticker].dropna().iloc[-1])
                notional = self.portfolio.capital * ticket.final_weight

                if notional > self.portfolio.cash:
                    continue  # Not enough cash

                pos = {
                    "trade_id": f"PT_{ticker}_{ticket.direction}_{today}",
                    "ticker": ticker,
                    "direction": ticket.direction,
                    "entry_date": today,
                    "entry_price": current_price,
                    "notional": notional,
                    "weight": ticket.final_weight,
                    "entry_z": ticket.entry_z,
                    "conviction": ticket.conviction_score,
                    "regime_at_entry": safety_label,
                    "current_price": current_price,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0,
                    "days_held": 0,
                    "current_z": ticket.entry_z,
                }
                self.portfolio.positions.append(pos)
                self.portfolio.cash -= notional
                existing_tickers.add(ticker)
                actions["entries"].append(pos["trade_id"])
                log.info("OPENED: %s %s @ $%.2f | Notional: $%.0f | Conv: %.3f",
                         ticket.direction, ticker, current_price, notional, ticket.conviction_score)

        # 4. Daily snapshot
        total_value = self.portfolio.cash + sum(
            p["notional"] + p["unrealized_pnl"] for p in self.portfolio.positions
        )
        self.portfolio.total_pnl = total_value - self.portfolio.capital
        self.portfolio.total_pnl_pct = self.portfolio.total_pnl / self.portfolio.capital

        # Performance stats
        closed = self.portfolio.closed_trades
        self.portfolio.n_trades = len(closed)
        self.portfolio.n_wins = sum(1 for t in closed if t["realized_pnl"] > 0)
        self.portfolio.win_rate = self.portfolio.n_wins / max(1, self.portfolio.n_trades)

        # Snapshot
        snapshot = {
            "date": today,
            "total_value": round(total_value, 2),
            "pnl_pct": round(self.portfolio.total_pnl_pct, 6),
            "n_positions": len(self.portfolio.positions),
            "cash": round(self.portfolio.cash, 2),
        }
        self.portfolio.daily_snapshots.append(snapshot)

        # Keep max 500 snapshots
        if len(self.portfolio.daily_snapshots) > 500:
            self.portfolio.daily_snapshots = self.portfolio.daily_snapshots[-500:]

        # Drawdown
        values = [s["total_value"] for s in self.portfolio.daily_snapshots]
        if values:
            peak = max(values)
            self.portfolio.max_drawdown = (min(values) - peak) / peak if peak > 0 else 0

        # Auto-journal: publish trades to audit trail + bus
        self._auto_journal(actions, today)

        return actions

    def _auto_journal(self, actions: dict, today: str) -> None:
        """Auto-publish trades to audit trail and agent bus."""
        try:
            from db.audit import AuditTrail
            audit = AuditTrail()
            for entry in actions.get("entries", []):
                audit.log_trade(today, f"PAPER-{entry['ticker']}", "OPEN", entry)
            for exit_t in actions.get("exits", []):
                audit.log_trade(today, f"PAPER-{exit_t['ticker']}", "CLOSE", exit_t)
        except Exception:
            pass  # Audit trail is optional

        try:
            from scripts.agent_bus import get_bus
            bus = get_bus()
            if actions.get("entries") or actions.get("exits"):
                bus.publish("paper_trader", {
                    "date": today,
                    "entries": len(actions.get("entries", [])),
                    "exits": len(actions.get("exits", [])),
                    "n_positions": len(self.portfolio.positions),
                    "total_pnl_pct": round(self.portfolio.total_pnl_pct, 4),
                    "win_rate": round(self.portfolio.win_rate, 3),
                })
        except Exception:
            pass  # Bus is optional

    # ── Status ───────────────────────────────────────────────────

    def status(self) -> str:
        """מחזיר סטטוס מפורט של הפורטפוליו."""
        p = self.portfolio
        lines = [
            f"{'='*60}",
            f"  PAPER TRADING PORTFOLIO — {date.today().isoformat()}",
            f"{'='*60}",
            f"",
            f"  Capital:     ${p.capital:>12,.0f}",
            f"  Total Value: ${p.capital + p.total_pnl:>12,.0f}",
            f"  Cash:        ${p.cash:>12,.0f}",
            f"  P&L:         ${p.total_pnl:>12,.0f} ({p.total_pnl_pct:+.2%})",
            f"  Max DD:      {p.max_drawdown:.2%}",
            f"  Trades:      {p.n_trades} ({p.n_wins} wins, {p.win_rate:.0%} WR)",
            f"",
        ]

        if p.positions:
            lines.append(f"  OPEN POSITIONS ({len(p.positions)}):")
            for pos in p.positions:
                pnl_str = f"${pos['unrealized_pnl']:>8,.0f} ({pos['unrealized_pnl_pct']:+.1%})"
                lines.append(
                    f"    {pos['direction']:<5} {pos['ticker']:<5} @ ${pos['entry_price']:.2f} "
                    f"→ ${pos['current_price']:.2f} | {pnl_str} | {pos['days_held']}d"
                )
        else:
            lines.append("  OPEN POSITIONS: none")

        if p.closed_trades:
            recent = p.closed_trades[-5:]
            lines.append(f"\n  RECENT TRADES (last {len(recent)}):")
            for t in reversed(recent):
                pnl_str = f"${t['realized_pnl']:>8,.0f} ({t['realized_pnl_pct']:+.1%})"
                lines.append(
                    f"    {t['direction']:<5} {t['ticker']:<5} | {t['entry_date']} → {t['exit_date']} "
                    f"| {pnl_str} | {t['exit_reason']}"
                )

        lines.extend(["", f"{'='*60}"])
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    trader = PaperTrader()
    trader.load()

    if "--reset" in sys.argv:
        trader.reset()
        print("Portfolio reset.")
        return

    if "--status" in sys.argv:
        print(trader.status())
        return

    if "--update" in sys.argv:
        # Load real data and run DSS
        from config.settings import get_settings
        from analytics.stat_arb import QuantEngine
        from analytics.signal_stack import SignalStackEngine
        from analytics.signal_regime_safety import compute_regime_safety_score
        from analytics.trade_structure import TradeStructureEngine, PositionSizingEngine

        settings = get_settings()
        engine = QuantEngine(settings)
        engine.load()
        master_df = engine.calculate_conviction_score()
        prices = engine.prices

        # DSS Signal Stack (simplified — no corr structure engine needed)
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

        print(f"\nActions: {len(actions['entries'])} entries, {len(actions['exits'])} exits, {len(actions['updates'])} updates")
        print(trader.status())
        return

    print("Usage: python analytics/paper_trader.py [--update|--status|--reset]")


if __name__ == "__main__":
    main()
