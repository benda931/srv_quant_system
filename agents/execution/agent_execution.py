"""
Execution Agent -- Signal-to-trade translator with paper execution.

Reads the latest signals from the pipeline, applies position sizing
from the leverage engine, and executes paper trades with realistic
slippage and commission simulation.

Flow:
1. Load latest master_df signals
2. Filter by conviction threshold (> 0.15)
3. Apply regime-based sizing (from leverage engine)
4. Check pre-trade risk (via Risk Guardian status)
5. Execute paper trades (via PaperTrader)
6. Log all executions with audit trail
7. Report summary: new entries, exits, P&L update

Slippage model: 5bps per trade (configurable)
Commission: $0.005/share (configurable)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# -- Root path ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # srv_quant_system/
sys.path.insert(0, str(ROOT))

# -- Logging ------------------------------------------------------------------
_LOG_DIR = ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            _LOG_DIR / "agent_execution.log",
            maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("agent_execution")

# -- Execution log output path ------------------------------------------------
EXECUTION_LOG_PATH = Path(__file__).resolve().parent / "execution_log.json"

# -- Risk status path (from Risk Guardian) ------------------------------------
RISK_STATUS_PATH = ROOT / "agents" / "risk_guardian" / "risk_status.json"

# -- Safe imports with fallbacks -----------------------------------------------
_IMPORTS_OK: Dict[str, bool] = {}

try:
    from analytics.paper_trader import PaperTrader
    _IMPORTS_OK["paper_trader"] = True
except ImportError as e:
    log.warning("Could not import PaperTrader: %s", e)
    _IMPORTS_OK["paper_trader"] = False

try:
    from analytics.leverage_engine import LeverageEngine
    _IMPORTS_OK["leverage_engine"] = True
except ImportError as e:
    log.warning("Could not import LeverageEngine: %s", e)
    _IMPORTS_OK["leverage_engine"] = False

try:
    from config.settings import get_settings, Settings
    _IMPORTS_OK["settings"] = True
except ImportError as e:
    log.warning("Could not import settings: %s", e)
    _IMPORTS_OK["settings"] = False

try:
    from agents.shared.agent_registry import get_registry, AgentStatus
    _IMPORTS_OK["registry"] = True
except ImportError as e:
    log.warning("Could not import agent_registry: %s", e)
    _IMPORTS_OK["registry"] = False

try:
    from scripts.agent_bus import get_bus
    _IMPORTS_OK["agent_bus"] = True
except ImportError as e:
    log.warning("Could not import agent_bus: %s", e)
    _IMPORTS_OK["agent_bus"] = False

try:
    from db.audit import AuditTrail
    _IMPORTS_OK["audit"] = True
except ImportError as e:
    log.warning("Could not import AuditTrail: %s", e)
    _IMPORTS_OK["audit"] = False


# =============================================================================
# Default execution parameters
# =============================================================================
_DEFAULT_CONVICTION_THRESHOLD = 0.15
_DEFAULT_SLIPPAGE_BPS = 5          # 5 basis points
_DEFAULT_COMMISSION_PER_SHARE = 0.005  # $0.005/share
_INITIAL_CAPITAL = 1_000_000.0


# =============================================================================
# ExecutionAgent
# =============================================================================
class ExecutionAgent:
    """
    Translates signals into paper trades with realistic simulation.

    Parameters
    ----------
    settings : optional
        A Settings object; loaded from config if not provided.
    slippage_bps : float
        Slippage per trade in basis points (default: 5).
    commission_per_share : float
        Commission per share in dollars (default: 0.005).
    """

    def __init__(
        self,
        settings: Optional[Any] = None,
        slippage_bps: float = _DEFAULT_SLIPPAGE_BPS,
        commission_per_share: float = _DEFAULT_COMMISSION_PER_SHARE,
    ) -> None:
        if settings is None and _IMPORTS_OK.get("settings"):
            try:
                self.settings = get_settings()
            except Exception:
                self.settings = None
        else:
            self.settings = settings

        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.conviction_threshold = _DEFAULT_CONVICTION_THRESHOLD

        self._leverage_engine: Optional[Any] = None
        if _IMPORTS_OK.get("leverage_engine"):
            try:
                max_lev = getattr(self.settings, "max_leverage", 5.0) if self.settings else 5.0
                self._leverage_engine = LeverageEngine(
                    base_capital=_INITIAL_CAPITAL,
                    max_leverage=max_lev,
                )
            except Exception as exc:
                log.warning("Failed to initialize LeverageEngine: %s", exc)

        self._execution_log: List[dict] = []

    # -----------------------------------------------------------------
    # Signal loading
    # -----------------------------------------------------------------

    def load_signals(self) -> pd.DataFrame:
        """
        Load master_df from latest pipeline run via QuantEngine or DuckDB.

        Returns an empty DataFrame if loading fails.
        """
        # Try QuantEngine first
        try:
            from analytics.stat_arb import QuantEngine
            settings = self.settings
            if settings is None and _IMPORTS_OK.get("settings"):
                settings = get_settings()
            engine = QuantEngine(settings)
            engine.load()
            master_df = engine.calculate_conviction_score()
            log.info("Loaded master_df via QuantEngine: %d rows", len(master_df))
            return master_df
        except Exception as exc:
            log.warning("QuantEngine load failed: %s", exc)

        # Fallback: try parquet
        try:
            parquet_path = ROOT / "data_lake" / "parquet" / "master_df.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                log.info("Loaded master_df from parquet: %d rows", len(df))
                return df
        except Exception as exc:
            log.warning("Parquet master_df load failed: %s", exc)

        log.warning("Could not load signals from any source")
        return pd.DataFrame()

    # -----------------------------------------------------------------
    # Risk Guardian check
    # -----------------------------------------------------------------

    def check_risk_guardian(self) -> bool:
        """
        Read risk_status.json and return False if level is RED or BLACK
        (meaning trading should be halted).
        """
        try:
            if RISK_STATUS_PATH.exists():
                data = json.loads(RISK_STATUS_PATH.read_text(encoding="utf-8"))
                level = data.get("level", "GREEN")
                log.info("Risk Guardian status: %s", level)
                if level in ("RED", "BLACK"):
                    log.warning(
                        "Risk Guardian level=%s -- trading HALTED", level
                    )
                    return False
                return True
            else:
                log.info(
                    "No risk_status.json found -- assuming GREEN (guardian not yet run)"
                )
                return True
        except Exception as exc:
            log.warning("Failed to read risk_status.json: %s -- proceeding cautiously", exc)
            return True

    # -----------------------------------------------------------------
    # Signal filtering
    # -----------------------------------------------------------------

    def filter_signals(self, master_df: pd.DataFrame) -> List[dict]:
        """
        Filter master_df by conviction threshold and direction != NEUTRAL.

        Returns a list of signal dicts with keys:
        ticker, direction, conviction, z_score, regime.
        """
        if master_df.empty:
            return []

        signals: List[dict] = []
        conviction_col = None
        for col in ("conviction_score", "conviction", "final_conviction"):
            if col in master_df.columns:
                conviction_col = col
                break

        if conviction_col is None:
            log.warning("No conviction column found in master_df")
            return []

        direction_col = None
        for col in ("direction", "signal_direction", "trade_direction"):
            if col in master_df.columns:
                direction_col = col
                break

        ticker_col = None
        for col in ("ticker", "sector_ticker", "symbol"):
            if col in master_df.columns:
                ticker_col = col
                break

        if ticker_col is None:
            log.warning("No ticker column found in master_df")
            return []

        for _, row in master_df.iterrows():
            try:
                conviction = float(row.get(conviction_col, 0))
                if not math.isfinite(conviction) or conviction < self.conviction_threshold:
                    continue

                direction = str(row.get(direction_col, "NEUTRAL")).upper() if direction_col else "NEUTRAL"
                if direction == "NEUTRAL":
                    continue

                signal = {
                    "ticker": str(row[ticker_col]),
                    "direction": direction,
                    "conviction": conviction,
                    "z_score": float(row.get("z_score", row.get("zscore", 0.0))),
                    "regime": str(row.get("regime", row.get("market_state", "NORMAL"))),
                }
                signals.append(signal)
            except Exception as exc:
                log.debug("Skipping row: %s", exc)
                continue

        # Sort by conviction descending
        signals.sort(key=lambda s: s["conviction"], reverse=True)
        log.info("Filtered %d signals above conviction threshold %.3f", len(signals), self.conviction_threshold)
        return signals

    # -----------------------------------------------------------------
    # Position sizing
    # -----------------------------------------------------------------

    def compute_sizing(self, signals: List[dict], regime: str = "NORMAL") -> List[dict]:
        """
        Apply leverage engine for position sizes.

        Adds 'notional' and 'shares' to each signal dict.
        """
        if not signals:
            return []

        capital = _INITIAL_CAPITAL

        # Get leverage target
        target_leverage = 1.0
        if self._leverage_engine is not None:
            try:
                leverage_result = self._leverage_engine.compute_target_leverage(
                    regime=regime,
                    vix=20.0,  # default; could read from market data
                    current_dd_pct=0.0,
                    strategy_sharpe=0.885,
                )
                target_leverage = max(leverage_result.target_leverage, 0.1)
                log.info(
                    "Leverage target: %.2fx (%s)", target_leverage, leverage_result.reasoning
                )
            except Exception as exc:
                log.warning("Leverage engine failed: %s -- using 1.0x", exc)

        max_positions = 8
        max_single_weight = 0.20
        sized: List[dict] = []

        for i, signal in enumerate(signals[:max_positions]):
            # Conviction-proportional weight, capped at max_single_weight
            raw_weight = min(signal["conviction"] * 0.4, max_single_weight)
            notional = capital * raw_weight * target_leverage

            signal_copy = dict(signal)
            signal_copy["weight"] = round(raw_weight, 4)
            signal_copy["notional"] = round(notional, 2)
            signal_copy["leverage_applied"] = round(target_leverage, 2)
            sized.append(signal_copy)

        log.info("Sized %d signals (leverage=%.2fx)", len(sized), target_leverage)
        return sized

    # -----------------------------------------------------------------
    # Slippage model
    # -----------------------------------------------------------------

    def apply_slippage(self, price: float, direction: str) -> float:
        """
        Add slippage to price.

        For BUY/LONG: price increases (worse fill).
        For SELL/SHORT: price decreases (worse fill).
        """
        slippage_pct = self.slippage_bps / 10_000.0
        if direction.upper() in ("LONG", "BUY"):
            return price * (1.0 + slippage_pct)
        else:
            return price * (1.0 - slippage_pct)

    # -----------------------------------------------------------------
    # Execute entries
    # -----------------------------------------------------------------

    def execute_entries(
        self, sized_signals: List[dict], prices: pd.DataFrame
    ) -> List[dict]:
        """
        Open new positions in PaperTrader based on sized signals.

        Returns a list of execution records.
        """
        if not sized_signals or not _IMPORTS_OK.get("paper_trader"):
            return []

        executions: List[dict] = []
        today = date.today().isoformat()

        try:
            trader = PaperTrader(settings=self.settings)
            trader.load()
        except Exception as exc:
            log.error("Failed to load PaperTrader: %s", exc)
            return []

        existing_tickers = {p["ticker"] for p in trader.portfolio.positions}

        for signal in sized_signals:
            ticker = signal["ticker"]
            if ticker in existing_tickers:
                log.info("Skipping %s -- already in portfolio", ticker)
                continue

            if ticker not in prices.columns:
                log.warning("No price data for %s -- skipping", ticker)
                continue

            if len(trader.portfolio.positions) >= 8:
                log.info("Max positions (8) reached -- stopping entries")
                break

            try:
                raw_price = float(prices[ticker].dropna().iloc[-1])
                fill_price = self.apply_slippage(raw_price, signal["direction"])
                notional = signal["notional"]

                # Commission estimate (assume ~100 shares equivalent)
                est_shares = notional / fill_price if fill_price > 0 else 0
                commission = est_shares * self.commission_per_share

                if notional > trader.portfolio.cash:
                    log.info("Insufficient cash for %s (need $%.0f, have $%.0f)",
                             ticker, notional, trader.portfolio.cash)
                    continue

                # Create position
                pos = {
                    "trade_id": f"EX_{ticker}_{signal['direction']}_{today}",
                    "ticker": ticker,
                    "direction": signal["direction"],
                    "entry_date": today,
                    "entry_price": round(fill_price, 4),
                    "notional": round(notional, 2),
                    "weight": signal.get("weight", 0.0),
                    "entry_z": signal.get("z_score", 0.0),
                    "conviction": signal["conviction"],
                    "regime_at_entry": signal.get("regime", "NORMAL"),
                    "current_price": round(fill_price, 4),
                    "unrealized_pnl": -round(commission, 2),  # start negative by commission
                    "unrealized_pnl_pct": 0.0,
                    "days_held": 0,
                    "current_z": signal.get("z_score", 0.0),
                }

                trader.portfolio.positions.append(pos)
                trader.portfolio.cash -= notional
                existing_tickers.add(ticker)

                execution_record = {
                    "action": "ENTRY",
                    "trade_id": pos["trade_id"],
                    "ticker": ticker,
                    "direction": signal["direction"],
                    "fill_price": fill_price,
                    "raw_price": raw_price,
                    "slippage_bps": self.slippage_bps,
                    "notional": notional,
                    "commission": round(commission, 2),
                    "conviction": signal["conviction"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                executions.append(execution_record)

                log.info(
                    "ENTRY: %s %s @ $%.2f (raw=$%.2f, slip=%dbps) | "
                    "Notional=$%.0f | Conv=%.3f",
                    signal["direction"], ticker, fill_price, raw_price,
                    self.slippage_bps, notional, signal["conviction"],
                )

                # Audit trail
                if _IMPORTS_OK.get("audit"):
                    try:
                        audit = AuditTrail()
                        audit.log_trade(today, f"EXEC-{ticker}", "OPEN", execution_record)
                    except Exception:
                        pass

            except Exception as exc:
                log.error("Failed to execute entry for %s: %s", ticker, exc)

        # Save portfolio
        try:
            trader.save()
        except Exception as exc:
            log.error("Failed to save portfolio after entries: %s", exc)

        return executions

    # -----------------------------------------------------------------
    # Execute exits
    # -----------------------------------------------------------------

    def execute_exits(self, prices: pd.DataFrame) -> List[dict]:
        """
        Check exit conditions and close positions.

        Returns a list of exit execution records.
        """
        if not _IMPORTS_OK.get("paper_trader"):
            return []

        exits: List[dict] = []
        today = date.today().isoformat()

        try:
            trader = PaperTrader(settings=self.settings)
            trader.load()
        except Exception as exc:
            log.error("Failed to load PaperTrader for exits: %s", exc)
            return []

        to_close: List[tuple] = []  # (pos, reason)

        for pos in trader.portfolio.positions:
            ticker = pos.get("ticker", "")
            if ticker not in prices.columns:
                continue

            try:
                current_price = float(prices[ticker].dropna().iloc[-1])
                entry_price = float(pos.get("entry_price", current_price))
                direction_sign = 1.0 if pos.get("direction") == "LONG" else -1.0
                pnl_pct = direction_sign * (current_price - entry_price) / entry_price
                days_held = (date.today() - date.fromisoformat(pos["entry_date"])).days

                # Update position
                pos["current_price"] = current_price
                pos["unrealized_pnl_pct"] = pnl_pct
                pos["unrealized_pnl"] = pnl_pct * float(pos.get("notional", 0))
                pos["days_held"] = days_held

                # Exit conditions
                reason = ""
                if days_held >= 25:
                    reason = "TIME_EXIT"
                elif pnl_pct >= 0.02:
                    reason = "PROFIT_TARGET"
                elif pnl_pct <= -0.03:
                    reason = "STOP_LOSS"

                if reason:
                    to_close.append((pos, reason))
            except Exception as exc:
                log.debug("Exit check failed for %s: %s", ticker, exc)

        for pos, reason in to_close:
            try:
                fill_price = self.apply_slippage(
                    pos["current_price"],
                    "SELL" if pos["direction"] == "LONG" else "BUY",
                )

                trade = {
                    "trade_id": pos["trade_id"],
                    "ticker": pos["ticker"],
                    "direction": pos["direction"],
                    "entry_date": pos["entry_date"],
                    "exit_date": today,
                    "entry_price": pos["entry_price"],
                    "exit_price": round(fill_price, 4),
                    "notional": pos["notional"],
                    "realized_pnl": round(pos["unrealized_pnl"], 2),
                    "realized_pnl_pct": round(pos["unrealized_pnl_pct"], 4),
                    "holding_days": pos["days_held"],
                    "exit_reason": reason,
                    "conviction": pos.get("conviction", 0.0),
                }

                trader.portfolio.closed_trades.append(trade)
                trader.portfolio.cash += pos["notional"] + pos["unrealized_pnl"]
                trader.portfolio.positions.remove(pos)

                exit_record = {
                    "action": "EXIT",
                    "trade_id": pos["trade_id"],
                    "ticker": pos["ticker"],
                    "direction": pos["direction"],
                    "fill_price": fill_price,
                    "realized_pnl": trade["realized_pnl"],
                    "realized_pnl_pct": trade["realized_pnl_pct"],
                    "exit_reason": reason,
                    "holding_days": trade["holding_days"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                exits.append(exit_record)

                log.info(
                    "EXIT: %s %s @ $%.2f | P&L: $%.0f (%.1f%%) | Reason: %s",
                    pos["direction"], pos["ticker"], fill_price,
                    trade["realized_pnl"], trade["realized_pnl_pct"] * 100, reason,
                )

                # Audit trail
                if _IMPORTS_OK.get("audit"):
                    try:
                        audit = AuditTrail()
                        audit.log_trade(today, f"EXEC-{pos['ticker']}", "CLOSE", exit_record)
                    except Exception:
                        pass

            except Exception as exc:
                log.error("Failed to execute exit for %s: %s", pos.get("ticker", "?"), exc)

        # Save portfolio
        try:
            trader.save()
        except Exception as exc:
            log.error("Failed to save portfolio after exits: %s", exc)

        return exits

    # -----------------------------------------------------------------
    # Save execution log
    # -----------------------------------------------------------------

    def _save_execution_log(self) -> None:
        """Save execution log to JSON."""
        try:
            EXECUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Load existing log
            existing: List[dict] = []
            if EXECUTION_LOG_PATH.exists():
                try:
                    existing = json.loads(EXECUTION_LOG_PATH.read_text(encoding="utf-8"))
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []

            existing.extend(self._execution_log)

            # Keep last 500 entries
            if len(existing) > 500:
                existing = existing[-500:]

            EXECUTION_LOG_PATH.write_text(
                json.dumps(existing, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("Saved %d execution records to %s", len(self._execution_log), EXECUTION_LOG_PATH)
        except Exception as exc:
            log.error("Failed to save execution log: %s", exc)

    # -----------------------------------------------------------------
    # Full run cycle
    # -----------------------------------------------------------------

    def run(self) -> dict:
        """
        Full cycle: signals -> risk check -> size -> execute -> report.

        Returns a summary dict.
        """
        started_at = datetime.now(timezone.utc)
        today = date.today().isoformat()
        self._execution_log = []

        # Registry heartbeat: RUNNING
        if _IMPORTS_OK.get("registry"):
            try:
                registry = get_registry()
                registry.register("execution_agent", role="signal-to-trade translator with paper execution")
                registry.heartbeat("execution_agent", AgentStatus.RUNNING)
            except Exception as exc:
                log.warning("Registry heartbeat failed: %s", exc)

        log.info("=" * 70)
        log.info("Execution Agent -- %s", today)
        log.info("=" * 70)

        summary: Dict[str, Any] = {
            "date": today,
            "started_at": started_at.isoformat(),
            "entries": [],
            "exits": [],
            "risk_guardian_ok": True,
            "signals_found": 0,
            "signals_sized": 0,
            "errors": [],
        }

        # Step 1: Check Risk Guardian
        risk_ok = self.check_risk_guardian()
        summary["risk_guardian_ok"] = risk_ok
        if not risk_ok:
            log.warning("Risk Guardian halted trading -- skipping execution")
            summary["errors"].append("Trading halted by Risk Guardian")
            self._finalize(summary)
            return summary

        # Step 2: Load signals
        try:
            master_df = self.load_signals()
        except Exception as exc:
            log.error("Failed to load signals: %s", exc)
            summary["errors"].append(f"Signal load failed: {exc}")
            self._finalize(summary)
            return summary

        if master_df.empty:
            log.info("No signals available -- nothing to execute")
            self._finalize(summary)
            return summary

        # Step 3: Filter signals
        signals = self.filter_signals(master_df)
        summary["signals_found"] = len(signals)

        if not signals:
            log.info("No actionable signals above threshold")
            self._finalize(summary)
            return summary

        # Determine regime from signals
        regime = signals[0].get("regime", "NORMAL")

        # Step 4: Size positions
        sized_signals = self.compute_sizing(signals, regime=regime)
        summary["signals_sized"] = len(sized_signals)

        # Step 5: Load prices
        prices = None
        try:
            prices_path = ROOT / "data_lake" / "parquet" / "prices.parquet"
            if prices_path.exists():
                prices = pd.read_parquet(prices_path)
            else:
                log.warning("No prices parquet found")
        except Exception as exc:
            log.error("Failed to load prices: %s", exc)

        if prices is None or prices.empty:
            summary["errors"].append("No price data available for execution")
            self._finalize(summary)
            return summary

        # Step 6: Execute exits first (free up capital)
        try:
            exit_records = self.execute_exits(prices)
            summary["exits"] = exit_records
            self._execution_log.extend(exit_records)
        except Exception as exc:
            log.error("Exit execution failed: %s", exc)
            summary["errors"].append(f"Exit execution error: {exc}")

        # Step 7: Execute entries
        try:
            entry_records = self.execute_entries(sized_signals, prices)
            summary["entries"] = entry_records
            self._execution_log.extend(entry_records)
        except Exception as exc:
            log.error("Entry execution failed: %s", exc)
            summary["errors"].append(f"Entry execution error: {exc}")

        # Step 8: Save execution log
        self._save_execution_log()

        # Step 9: Publish to bus
        if _IMPORTS_OK.get("agent_bus"):
            try:
                bus = get_bus()
                bus.publish("execution_agent", {
                    "date": today,
                    "entries": len(summary["entries"]),
                    "exits": len(summary["exits"]),
                    "signals_found": summary["signals_found"],
                    "signals_sized": summary["signals_sized"],
                    "risk_guardian_ok": summary["risk_guardian_ok"],
                })
            except Exception as exc:
                log.warning("Failed to publish to agent_bus: %s", exc)

        self._finalize(summary)
        return summary

    def _finalize(self, summary: dict) -> None:
        """Finalize run: update registry heartbeat."""
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()

        log.info(
            "Execution summary: %d entries, %d exits, %d signals, %d errors",
            len(summary.get("entries", [])),
            len(summary.get("exits", [])),
            summary.get("signals_found", 0),
            len(summary.get("errors", [])),
        )

        if _IMPORTS_OK.get("registry"):
            try:
                registry = get_registry()
                has_errors = bool(summary.get("errors"))
                if has_errors:
                    registry.heartbeat(
                        "execution_agent", AgentStatus.COMPLETED,
                        error="; ".join(summary["errors"][:3]),
                    )
                else:
                    registry.heartbeat("execution_agent", AgentStatus.COMPLETED)
            except Exception as exc:
                log.warning("Registry heartbeat failed: %s", exc)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Execution Agent")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument(
        "--interval", type=int, default=600,
        help="Seconds between runs in loop mode (default: 600)",
    )
    args = parser.parse_args()

    agent = ExecutionAgent()

    if args.once:
        summary = agent.run()
        log.info(
            "Execution Agent completed: %d entries, %d exits",
            len(summary.get("entries", [])), len(summary.get("exits", [])),
        )
        return

    # Loop mode
    log.info("Execution Agent starting in loop mode (interval=%ds)", args.interval)
    while True:
        try:
            summary = agent.run()
            log.info(
                "Cycle complete: %d entries, %d exits",
                len(summary.get("entries", [])), len(summary.get("exits", [])),
            )
        except Exception as exc:
            log.error("Execution Agent cycle failed: %s", exc)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
