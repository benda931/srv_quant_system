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
import copy
import json
import logging
import math
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
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

# -- Institutional input paths ------------------------------------------------
_PORTFOLIO_WEIGHTS_PATH = ROOT / "agents" / "portfolio_construction" / "portfolio_weights.json"
_RISK_STATUS_PATH = ROOT / "agents" / "risk_guardian" / "risk_status.json"
_REGIME_FORECAST_PATH = ROOT / "agents" / "regime_forecaster" / "regime_forecast.json"
_DECAY_STATUS_PATH = ROOT / "agents" / "alpha_decay" / "decay_status.json"

# -- Execution style constants ------------------------------------------------
_DUST_THRESHOLD_WEIGHT = 0.005  # below this delta weight -> NO_TRADE
_PROFIT_TAKE_PCT = 0.03
_STOP_LOSS_PCT = -0.04
_TIME_STOP_DAYS = 25
_MAX_PENDING_CYCLES = 5

# -- Slippage profiles per execution style ------------------------------------
_SLIPPAGE_PROFILES: Dict[str, Dict[str, float]] = {
    "AGGRESSIVE":      {"fill_pct": 1.00, "slippage_bps": 8.0},
    "PASSIVE":         {"fill_pct": 0.80, "slippage_bps": 3.0},
    "TWAP":            {"fill_pct": 0.95, "slippage_bps": 5.0},
    "SCALED_ENTRY":    {"fill_pct": 0.60, "slippage_bps": 4.0},
    "DEFENSIVE_EXIT":  {"fill_pct": 0.90, "slippage_bps": 6.0},
    "EMERGENCY_EXIT":  {"fill_pct": 1.00, "slippage_bps": 15.0},
    "NO_TRADE":        {"fill_pct": 0.00, "slippage_bps": 0.0},
    "DEFERRED":        {"fill_pct": 0.00, "slippage_bps": 0.0},
}


# =============================================================================
# OrderIntent dataclass
# =============================================================================
@dataclass
class OrderIntent:
    """Represents a single order intention with full institutional context."""

    order_id: str = ""
    ticker: str = ""
    side: str = ""                      # BUY / SELL
    order_type: str = ""                # ENTRY / EXIT / INCREASE / DECREASE / FLIP / HOLD
    target_weight: float = 0.0
    current_weight: float = 0.0
    delta_notional: float = 0.0
    urgency: str = "MEDIUM"             # HIGH / MEDIUM / LOW / DEFERRED
    execution_style: str = "PASSIVE"    # AGGRESSIVE / PASSIVE / TWAP / SCALED_ENTRY / DEFENSIVE_EXIT / EMERGENCY_EXIT / NO_TRADE
    allowed: bool = True
    restrictions: List[str] = field(default_factory=list)
    expected_cost_bps: float = 0.0
    expected_impact_bps: float = 0.0
    sleeve_role: str = "CORE"           # CORE / TACTICAL / etc
    health_state: str = "HEALTHY"
    regime_eligible: bool = True
    fill_status: str = "PENDING"        # PENDING / PARTIAL / FILLED / REJECTED / DEFERRED
    filled_pct: float = 0.0
    reason: str = ""
    fill_price: float = 0.0
    decision_price: float = 0.0
    arrival_price: float = 0.0
    cycle_age: int = 0                  # how many cycles this intent has been pending

    def to_dict(self) -> dict:
        return asdict(self)


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
        self._pending_orders: List[OrderIntent] = []

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
    # Professional-grade execution analytics
    # -----------------------------------------------------------------

    def simulate_twap(
        self, ticker: str, target_shares: float, duration_minutes: int = 30
    ) -> dict:
        """
        Simulate Time-Weighted Average Price (TWAP) execution.

        Splits the order into equal slices over the given duration and adds
        random slippage per slice (normal distribution, mean=0, std=2bps).

        Parameters
        ----------
        ticker : str
            The ticker symbol.
        target_shares : float
            Total shares to execute.
        duration_minutes : int
            Duration to spread the order over (default: 30).

        Returns
        -------
        dict
            avg_price, total_slippage_bps, execution_quality, n_slices,
            detail
        """
        result = {
            "ticker": ticker,
            "target_shares": target_shares,
            "avg_price": None,
            "total_slippage_bps": None,
            "execution_quality": None,
            "n_slices": 0,
            "detail": "",
        }
        try:
            # Load reference price
            prices_path = ROOT / "data_lake" / "parquet" / "prices.parquet"
            if not prices_path.exists():
                result["detail"] = "No price data for TWAP simulation"
                return result

            prices = pd.read_parquet(prices_path)
            if ticker not in prices.columns:
                result["detail"] = f"No price data for {ticker}"
                return result

            ref_price = float(prices[ticker].dropna().iloc[-1])

            # Split into 1-minute slices
            n_slices = max(1, duration_minutes)
            shares_per_slice = target_shares / n_slices

            np.random.seed(hash(ticker) % 2**31)
            slippage_per_slice = np.random.normal(0, 2, n_slices)  # 2bps std

            fill_prices = []
            for i in range(n_slices):
                slip = slippage_per_slice[i] / 10_000.0
                fill_price = ref_price * (1.0 + slip)
                fill_prices.append(fill_price)

            avg_price = float(np.mean(fill_prices))
            total_slippage_bps = float((avg_price / ref_price - 1.0) * 10_000)

            # Execution quality: 1.0 = perfect, lower = worse
            # Quality = 1 - |slippage| / 10bps (normalize against 10bps threshold)
            execution_quality = max(0.0, 1.0 - abs(total_slippage_bps) / 10.0)

            result.update({
                "avg_price": round(avg_price, 4),
                "ref_price": round(ref_price, 4),
                "total_slippage_bps": round(total_slippage_bps, 2),
                "execution_quality": round(execution_quality, 4),
                "n_slices": n_slices,
                "detail": (
                    f"TWAP {ticker}: {n_slices} slices over {duration_minutes}min, "
                    f"avg={avg_price:.4f} vs ref={ref_price:.4f}, "
                    f"slip={total_slippage_bps:.2f}bps, quality={execution_quality:.3f}"
                ),
            })
            log.info(result["detail"])
        except Exception as exc:
            log.error("TWAP simulation failed for %s: %s", ticker, exc)
            result["detail"] = f"TWAP error: {exc}"

        return result

    def estimate_market_impact(self, ticker: str, notional: float) -> float:
        """
        Estimate market impact using the Almgren-Chriss model.

        impact = sigma * sqrt(notional / ADV) * eta

        Parameters
        ----------
        ticker : str
            The ticker symbol.
        notional : float
            Dollar value of the order.

        Returns
        -------
        float
            Estimated market impact in basis points. Returns 0.0 on error.
        """
        try:
            prices_path = ROOT / "data_lake" / "parquet" / "prices.parquet"
            if not prices_path.exists():
                log.warning("No price data for market impact estimation")
                return 0.0

            prices = pd.read_parquet(prices_path)
            if ticker not in prices.columns:
                log.warning("No price data for %s in market impact", ticker)
                return 0.0

            px = prices[ticker].dropna()
            if len(px) < 30:
                return 0.0

            # Realized volatility (annualized)
            rets = np.log(px / px.shift(1)).dropna()
            sigma = float(rets.iloc[-20:].std() * np.sqrt(252))

            # Estimate ADV from price level and vol
            current_price = float(px.iloc[-1])
            daily_vol = float(rets.iloc[-20:].std())
            estimated_adv = current_price * max(daily_vol, 0.005) * 1e6

            # Almgren-Chriss permanent impact
            eta = 0.1  # permanent impact coefficient
            participation_rate = notional / estimated_adv if estimated_adv > 0 else 1.0
            impact_bps = sigma * math.sqrt(max(participation_rate, 0)) * eta * 10_000

            log.info(
                "Market impact %s: %.1f bps (notional=$%.0f, ADV=$%.0f, sigma=%.3f)",
                ticker, impact_bps, notional, estimated_adv, sigma,
            )
            return round(impact_bps, 2)
        except Exception as exc:
            log.error("Market impact estimation failed for %s: %s", ticker, exc)
            return 0.0

    def optimal_entry_timing(self, ticker: str, direction: str) -> dict:
        """
        Analyze optimal entry timing based on intraday patterns and VIX state.

        Checks day-of-week effects and whether VIX is mean-reverting or
        trending to recommend urgency and delay.

        Parameters
        ----------
        ticker : str
            The ticker symbol.
        direction : str
            Trade direction (LONG/SHORT).

        Returns
        -------
        dict
            urgency (HIGH/MEDIUM/LOW), recommended_delay_hours, reason, detail
        """
        result = {
            "ticker": ticker,
            "direction": direction,
            "urgency": "MEDIUM",
            "recommended_delay_hours": 0,
            "reason": "",
            "detail": "",
        }
        try:
            prices_path = ROOT / "data_lake" / "parquet" / "prices.parquet"
            if not prices_path.exists():
                result["reason"] = "No price data; defaulting to MEDIUM urgency"
                result["detail"] = result["reason"]
                return result

            prices = pd.read_parquet(prices_path)
            reasons = []

            # Check VIX state
            vix = prices.get("^VIX", pd.Series(dtype=float)).dropna()
            if len(vix) >= 20:
                current_vix = float(vix.iloc[-1])
                vix_ma20 = float(vix.iloc[-20:].mean())
                vix_trend = current_vix - vix_ma20

                if current_vix > 30:
                    if direction.upper() in ("LONG", "BUY"):
                        result["urgency"] = "LOW"
                        result["recommended_delay_hours"] = 24
                        reasons.append(f"VIX elevated ({current_vix:.1f}), wait for mean reversion")
                    else:
                        result["urgency"] = "HIGH"
                        result["recommended_delay_hours"] = 0
                        reasons.append(f"VIX elevated ({current_vix:.1f}), SHORT entry favored now")
                elif vix_trend > 3:
                    result["urgency"] = "LOW"
                    result["recommended_delay_hours"] = 12
                    reasons.append(f"VIX trending up ({vix_trend:+.1f} vs MA20), delay entry")
                elif vix_trend < -3:
                    result["urgency"] = "HIGH"
                    result["recommended_delay_hours"] = 0
                    reasons.append(f"VIX trending down ({vix_trend:+.1f} vs MA20), favorable entry")

            # Day-of-week effect
            if len(prices.index) > 0:
                today_dow = pd.Timestamp.now().dayofweek  # 0=Mon, 4=Fri
                if today_dow == 0:  # Monday
                    reasons.append("Monday: historically higher open volatility")
                    if result["urgency"] != "HIGH":
                        result["recommended_delay_hours"] = max(
                            result["recommended_delay_hours"], 2
                        )
                elif today_dow == 4:  # Friday
                    reasons.append("Friday: weekend risk, consider partial sizing")
                    result["recommended_delay_hours"] = max(
                        result["recommended_delay_hours"], 0
                    )

            # Check if ticker is in a momentum state
            if ticker in prices.columns:
                px = prices[ticker].dropna()
                if len(px) >= 20:
                    rets_5d = float(px.iloc[-1] / px.iloc[-6] - 1) if len(px) >= 6 else 0
                    if direction.upper() in ("LONG", "BUY") and rets_5d < -0.05:
                        result["urgency"] = "HIGH"
                        result["recommended_delay_hours"] = 0
                        reasons.append(f"{ticker} oversold (5d ret={rets_5d:.1%}), mean reversion entry")
                    elif direction.upper() in ("SHORT", "SELL") and rets_5d > 0.05:
                        result["urgency"] = "HIGH"
                        result["recommended_delay_hours"] = 0
                        reasons.append(f"{ticker} overbought (5d ret={rets_5d:.1%}), short entry")

            if not reasons:
                reasons.append("No strong timing signal; standard entry")

            result["reason"] = "; ".join(reasons)
            result["detail"] = (
                f"Entry timing {ticker} {direction}: urgency={result['urgency']}, "
                f"delay={result['recommended_delay_hours']}h, reasons={result['reason']}"
            )
            log.info(result["detail"])
        except Exception as exc:
            log.error("Optimal entry timing failed for %s: %s", ticker, exc)
            result["detail"] = f"Timing error: {exc}"
            result["reason"] = f"Error: {exc}"

        return result

    def reconcile_portfolio(self) -> dict:
        """
        Compare paper portfolio positions vs expected positions from signals.

        Identifies missing entries (signal but no position), stale positions
        (position but no active signal), and size mismatches.

        Returns
        -------
        dict
            missing_entries, stale_positions, size_mismatches, n_issues, detail
        """
        result = {
            "missing_entries": [],
            "stale_positions": [],
            "size_mismatches": [],
            "n_issues": 0,
            "detail": "",
        }
        try:
            # Load current signals
            try:
                signals_df = self.load_signals()
                signals = self.filter_signals(signals_df)
            except Exception:
                signals = []

            # Load current portfolio
            portfolio_path = ROOT / "data" / "paper_portfolio.json"
            positions = []
            if portfolio_path.exists():
                try:
                    pdata = json.loads(portfolio_path.read_text(encoding="utf-8"))
                    positions = pdata.get("positions", [])
                except Exception:
                    pass

            signal_tickers = {s["ticker"]: s for s in signals}
            position_tickers = {p["ticker"]: p for p in positions}

            # Missing entries: signal exists but no position
            for ticker, sig in signal_tickers.items():
                if ticker not in position_tickers:
                    result["missing_entries"].append({
                        "ticker": ticker,
                        "direction": sig["direction"],
                        "conviction": sig["conviction"],
                    })

            # Stale positions: position exists but no current signal
            for ticker, pos in position_tickers.items():
                if ticker not in signal_tickers:
                    result["stale_positions"].append({
                        "ticker": ticker,
                        "direction": pos.get("direction", "UNKNOWN"),
                        "days_held": pos.get("days_held", 0),
                        "unrealized_pnl_pct": pos.get("unrealized_pnl_pct", 0),
                    })

            # Size mismatches: direction mismatch
            for ticker in set(signal_tickers) & set(position_tickers):
                sig_dir = signal_tickers[ticker]["direction"].upper()
                pos_dir = position_tickers[ticker].get("direction", "").upper()
                if sig_dir != pos_dir:
                    result["size_mismatches"].append({
                        "ticker": ticker,
                        "signal_direction": sig_dir,
                        "position_direction": pos_dir,
                        "issue": "Direction mismatch",
                    })

            n_issues = (
                len(result["missing_entries"]) +
                len(result["stale_positions"]) +
                len(result["size_mismatches"])
            )
            result["n_issues"] = n_issues
            result["detail"] = (
                f"Reconciliation: {n_issues} issues "
                f"({len(result['missing_entries'])} missing, "
                f"{len(result['stale_positions'])} stale, "
                f"{len(result['size_mismatches'])} mismatches)"
            )
            log.info(result["detail"])
        except Exception as exc:
            log.error("Portfolio reconciliation failed: %s", exc)
            result["detail"] = f"Reconciliation error: {exc}"

        return result

    def compute_execution_quality(self) -> dict:
        """
        Analyze execution quality for closed trades.

        For each closed trade, computes slippage vs theoretical price,
        implementation shortfall, and timing cost.

        Returns
        -------
        dict
            avg_slippage_bps, implementation_shortfall_bps, timing_cost_bps,
            n_trades_analyzed, per_trade (list), detail
        """
        result = {
            "avg_slippage_bps": None,
            "implementation_shortfall_bps": None,
            "timing_cost_bps": None,
            "n_trades_analyzed": 0,
            "per_trade": [],
            "detail": "",
        }
        try:
            # Load execution log
            if not EXECUTION_LOG_PATH.exists():
                result["detail"] = "No execution log for quality analysis"
                return result

            log_data = json.loads(EXECUTION_LOG_PATH.read_text(encoding="utf-8"))
            if not isinstance(log_data, list):
                result["detail"] = "Invalid execution log format"
                return result

            # Filter to entries that have both raw_price and fill_price
            entries = [
                e for e in log_data
                if e.get("action") == "ENTRY"
                and e.get("raw_price") is not None
                and e.get("fill_price") is not None
            ]

            if not entries:
                result["detail"] = "No entry trades with price data for quality analysis"
                return result

            slippages = []
            shortfalls = []
            per_trade = []

            for entry in entries:
                raw = float(entry["raw_price"])
                fill = float(entry["fill_price"])
                direction = entry.get("direction", "LONG").upper()

                if raw <= 0:
                    continue

                # Slippage in bps
                if direction in ("LONG", "BUY"):
                    slippage_bps = (fill / raw - 1.0) * 10_000
                else:
                    slippage_bps = (1.0 - fill / raw) * 10_000

                # Implementation shortfall: same as slippage for paper trades
                shortfall_bps = abs(slippage_bps)

                slippages.append(slippage_bps)
                shortfalls.append(shortfall_bps)

                per_trade.append({
                    "trade_id": entry.get("trade_id", ""),
                    "ticker": entry.get("ticker", ""),
                    "slippage_bps": round(slippage_bps, 2),
                    "shortfall_bps": round(shortfall_bps, 2),
                })

            if not slippages:
                result["detail"] = "No valid trades for quality computation"
                return result

            avg_slip = float(np.mean(slippages))
            avg_shortfall = float(np.mean(shortfalls))
            # Timing cost: std of slippage (variance from optimal)
            timing_cost = float(np.std(slippages))

            result.update({
                "avg_slippage_bps": round(avg_slip, 2),
                "implementation_shortfall_bps": round(avg_shortfall, 2),
                "timing_cost_bps": round(timing_cost, 2),
                "n_trades_analyzed": len(slippages),
                "per_trade": per_trade[-20:],  # Last 20 trades
                "detail": (
                    f"Execution quality: {len(slippages)} trades, "
                    f"avg slip={avg_slip:.2f}bps, shortfall={avg_shortfall:.2f}bps, "
                    f"timing cost={timing_cost:.2f}bps"
                ),
            })
            log.info(result["detail"])
        except Exception as exc:
            log.error("Execution quality computation failed: %s", exc)
            result["detail"] = f"Quality analysis error: {exc}"

        return result

    # =================================================================
    # INSTITUTIONAL EXECUTION DESK — added methods
    # =================================================================

    # -----------------------------------------------------------------
    # 1. ExecutionInputAssembler
    # -----------------------------------------------------------------

    def assemble_execution_inputs(self) -> Dict[str, Any]:
        """
        Load all upstream agent outputs needed for institutional execution.

        Returns structured dict with available_inputs list and each
        agent's data under its own key.
        """
        inputs: Dict[str, Any] = {
            "available_inputs": [],
            "portfolio_construction": None,
            "risk_guardian": None,
            "regime_forecast": None,
            "alpha_decay": None,
        }

        # Portfolio Construction targets
        try:
            if _PORTFOLIO_WEIGHTS_PATH.exists():
                data = json.loads(_PORTFOLIO_WEIGHTS_PATH.read_text(encoding="utf-8"))
                inputs["portfolio_construction"] = data
                inputs["available_inputs"].append("portfolio_construction")
                log.info("Loaded portfolio construction targets: %d positions",
                         len(data.get("weights", {})))
        except Exception as exc:
            log.warning("Failed to load portfolio_weights.json: %s", exc)

        # Risk Guardian veto
        try:
            if _RISK_STATUS_PATH.exists():
                data = json.loads(_RISK_STATUS_PATH.read_text(encoding="utf-8"))
                inputs["risk_guardian"] = data
                inputs["available_inputs"].append("risk_guardian")
                log.info("Loaded risk guardian status: level=%s",
                         data.get("level", "UNKNOWN"))
        except Exception as exc:
            log.warning("Failed to load risk_status.json: %s", exc)

        # Regime Forecast
        try:
            if _REGIME_FORECAST_PATH.exists():
                data = json.loads(_REGIME_FORECAST_PATH.read_text(encoding="utf-8"))
                inputs["regime_forecast"] = data
                inputs["available_inputs"].append("regime_forecast")
                log.info("Loaded regime forecast: predicted=%s",
                         data.get("predicted_regime", "UNKNOWN"))
        except Exception as exc:
            log.warning("Failed to load regime_forecast.json: %s", exc)

        # Alpha Decay
        try:
            if _DECAY_STATUS_PATH.exists():
                data = json.loads(_DECAY_STATUS_PATH.read_text(encoding="utf-8"))
                inputs["alpha_decay"] = data
                inputs["available_inputs"].append("alpha_decay")
                log.info("Loaded alpha decay status: level=%s",
                         data.get("decay_level", "UNKNOWN"))
        except Exception as exc:
            log.warning("Failed to load decay_status.json: %s", exc)

        log.info("Assembled execution inputs: %s", inputs["available_inputs"])
        return inputs

    # -----------------------------------------------------------------
    # 2. Portfolio Layer Separation
    # -----------------------------------------------------------------

    def compute_portfolio_layers(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute four portfolio layers from desired to realized.

        Returns dict with desired_portfolio, risk_approved_portfolio,
        executable_portfolio, realized_portfolio — each mapping ticker to
        a dict of target_weight, current_weight, delta, order_type.
        """
        layers: Dict[str, Any] = {
            "desired_portfolio": {},
            "risk_approved_portfolio": {},
            "executable_portfolio": {},
            "realized_portfolio": {},
        }

        # --- Desired portfolio: from Portfolio Construction ---
        pc = inputs.get("portfolio_construction") or {}
        desired_weights: Dict[str, float] = pc.get("weights", {})

        # --- Realized portfolio: current positions ---
        realized_weights: Dict[str, float] = {}
        try:
            portfolio_path = ROOT / "data" / "paper_portfolio.json"
            if portfolio_path.exists():
                pdata = json.loads(portfolio_path.read_text(encoding="utf-8"))
                positions = pdata.get("positions", [])
                total_value = float(pdata.get("total_value", _INITIAL_CAPITAL))
                if total_value <= 0:
                    total_value = _INITIAL_CAPITAL
                for pos in positions:
                    ticker = pos.get("ticker", "")
                    notional = float(pos.get("notional", 0))
                    realized_weights[ticker] = notional / total_value
        except Exception as exc:
            log.warning("Failed to load realized portfolio: %s", exc)

        # All tickers across desired + realized
        all_tickers = set(desired_weights.keys()) | set(realized_weights.keys())

        # --- Risk Guardian veto/haircut ---
        rg = inputs.get("risk_guardian") or {}
        veto = rg.get("veto", {})
        can_add_new = veto.get("can_allocate_new_risk", True)
        must_reduce = veto.get("must_reduce_existing_risk", False)
        emergency_unwind = veto.get("emergency_unwind_required", False)
        vetoed_sleeves = set(veto.get("vetoed_sleeves", []))
        haircut_required = rg.get("proposed_assessment", {}).get("haircut_required", False)
        haircut_factor = 0.8 if haircut_required else 1.0

        # --- Regime constraints ---
        regime_data = inputs.get("regime_forecast") or {}
        regime = regime_data.get("predicted_regime", "NORMAL")

        # Regime-based turnover cap
        regime_turnover_cap = {"CALM": 0.30, "NORMAL": 0.25, "TENSION": 0.15, "CRISIS": 0.08}
        max_turnover = regime_turnover_cap.get(regime, 0.20)

        for ticker in all_tickers:
            target_w = desired_weights.get(ticker, 0.0)
            current_w = realized_weights.get(ticker, 0.0)
            delta = target_w - current_w

            # Determine order type
            if current_w == 0 and target_w > 0:
                order_type = "ENTRY"
            elif current_w > 0 and target_w == 0:
                order_type = "EXIT"
            elif delta > 0:
                order_type = "INCREASE"
            elif delta < 0:
                order_type = "DECREASE"
            else:
                order_type = "HOLD"

            row = {
                "target_weight": round(target_w, 6),
                "current_weight": round(current_w, 6),
                "delta": round(delta, 6),
                "order_type": order_type,
            }

            # Layer 1: Desired
            layers["desired_portfolio"][ticker] = dict(row)

            # Layer 2: Risk-approved (apply haircuts/vetoes)
            approved_row = dict(row)
            if ticker in vetoed_sleeves:
                approved_row["target_weight"] = approved_row["current_weight"]
                approved_row["delta"] = 0.0
                approved_row["order_type"] = "HOLD"
            elif emergency_unwind:
                approved_row["target_weight"] = 0.0
                approved_row["delta"] = -approved_row["current_weight"]
                approved_row["order_type"] = "EXIT" if approved_row["current_weight"] > 0 else "HOLD"
            elif not can_add_new and order_type in ("ENTRY", "INCREASE"):
                approved_row["target_weight"] = approved_row["current_weight"]
                approved_row["delta"] = 0.0
                approved_row["order_type"] = "HOLD"
            elif must_reduce and order_type in ("ENTRY", "INCREASE", "HOLD"):
                approved_row["target_weight"] = round(approved_row["current_weight"] * 0.85, 6)
                approved_row["delta"] = round(approved_row["target_weight"] - approved_row["current_weight"], 6)
                approved_row["order_type"] = "DECREASE" if approved_row["delta"] < 0 else "HOLD"
            elif haircut_required:
                approved_row["target_weight"] = round(target_w * haircut_factor, 6)
                approved_row["delta"] = round(approved_row["target_weight"] - current_w, 6)
            layers["risk_approved_portfolio"][ticker] = approved_row

            # Layer 3: Executable (apply regime/turnover/dust constraints)
            exec_row = dict(approved_row)
            if abs(exec_row["delta"]) < _DUST_THRESHOLD_WEIGHT:
                exec_row["delta"] = 0.0
                exec_row["target_weight"] = exec_row["current_weight"]
                exec_row["order_type"] = "HOLD"
            layers["executable_portfolio"][ticker] = exec_row

            # Layer 4: Realized (current state)
            layers["realized_portfolio"][ticker] = {
                "target_weight": round(current_w, 6),
                "current_weight": round(current_w, 6),
                "delta": 0.0,
                "order_type": "HOLD",
            }

        log.info(
            "Portfolio layers computed: %d tickers, desired=%d, executable_deltas=%d",
            len(all_tickers),
            len(desired_weights),
            sum(1 for t in layers["executable_portfolio"].values() if t["delta"] != 0),
        )
        return layers

    # -----------------------------------------------------------------
    # 3. OrderIntent builder
    # -----------------------------------------------------------------

    def build_order_intents(
        self, layers: Dict[str, Any], inputs: Dict[str, Any]
    ) -> List[OrderIntent]:
        """
        Build OrderIntent objects for every ticker with a non-zero executable delta.

        Applies governance: veto, decay, regime eligibility, execution style.
        """
        intents: List[OrderIntent] = []
        executable = layers.get("executable_portfolio", {})

        rg = inputs.get("risk_guardian") or {}
        veto = rg.get("veto", {})
        vetoed_sleeves = set(veto.get("vetoed_sleeves", []))
        emergency_unwind = veto.get("emergency_unwind_required", False)

        decay_data = inputs.get("alpha_decay") or {}
        decay_level = decay_data.get("decay_level", "HEALTHY")

        regime_data = inputs.get("regime_forecast") or {}
        regime = regime_data.get("predicted_regime", "NORMAL")
        transition_prob = regime_data.get("transition_probability", 0.0)
        # Derive transition state
        if transition_prob > 0.6:
            transition_state = "ACTIVE_TRANSITION"
        elif transition_prob > 0.4:
            transition_state = "EARLY_WARNING"
        else:
            transition_state = "STABLE"

        for ticker, row in executable.items():
            delta = row.get("delta", 0.0)
            order_type = row.get("order_type", "HOLD")
            if order_type == "HOLD" and abs(delta) < _DUST_THRESHOLD_WEIGHT:
                continue

            target_w = row.get("target_weight", 0.0)
            current_w = row.get("current_weight", 0.0)
            side = "BUY" if delta > 0 else "SELL"
            delta_notional = abs(delta) * _INITIAL_CAPITAL

            # Governance checks
            allowed = True
            restrictions: List[str] = []

            if ticker in vetoed_sleeves:
                allowed = False
                restrictions.append("VETOED_BY_RISK_GUARDIAN")

            if decay_level in ("STRUCTURAL_DECAY", "DEAD") and order_type in ("ENTRY", "INCREASE"):
                allowed = False
                restrictions.append(f"BLOCKED_BY_DECAY_{decay_level}")

            regime_eligible = True
            if regime == "CRISIS" and order_type == "ENTRY":
                regime_eligible = False
                restrictions.append("CRISIS_REGIME_ENTRY_BLOCKED")
                allowed = False

            # Urgency
            if emergency_unwind:
                urgency = "HIGH"
            elif order_type == "EXIT":
                urgency = "HIGH"
            elif regime == "CRISIS":
                urgency = "LOW"
            elif abs(delta) > 0.05:
                urgency = "HIGH"
            elif abs(delta) > 0.02:
                urgency = "MEDIUM"
            else:
                urgency = "LOW"

            # Execution style
            execution_style = self.determine_execution_style(
                None, regime, transition_state, urgency, order_type, emergency_unwind
            )

            profile = _SLIPPAGE_PROFILES.get(execution_style, _SLIPPAGE_PROFILES["PASSIVE"])

            intent = OrderIntent(
                order_id=f"OI_{ticker}_{uuid.uuid4().hex[:8]}",
                ticker=ticker,
                side=side,
                order_type=order_type,
                target_weight=target_w,
                current_weight=current_w,
                delta_notional=round(delta_notional, 2),
                urgency=urgency,
                execution_style=execution_style,
                allowed=allowed,
                restrictions=restrictions,
                expected_cost_bps=profile["slippage_bps"],
                expected_impact_bps=round(profile["slippage_bps"] * 0.6, 2),
                sleeve_role="CORE",
                health_state=decay_level,
                regime_eligible=regime_eligible,
                fill_status="PENDING" if allowed else "REJECTED",
                filled_pct=0.0,
                reason="; ".join(restrictions) if restrictions else order_type,
            )
            intents.append(intent)

        log.info(
            "Built %d order intents (%d allowed, %d rejected)",
            len(intents),
            sum(1 for i in intents if i.allowed),
            sum(1 for i in intents if not i.allowed),
        )
        return intents

    # -----------------------------------------------------------------
    # 4. ExecutionPolicyEngine
    # -----------------------------------------------------------------

    def determine_execution_style(
        self,
        intent: Optional[Any],
        regime: str,
        transition_state: str,
        urgency: str,
        order_type: str = "",
        emergency: bool = False,
    ) -> str:
        """
        Determine the execution style based on regime, urgency, and order type.

        Returns one of: AGGRESSIVE, PASSIVE, TWAP, SCALED_ENTRY,
        DEFENSIVE_EXIT, EMERGENCY_EXIT, NO_TRADE, DEFERRED.
        """
        if emergency:
            return "EMERGENCY_EXIT"

        otype = order_type or (intent.order_type if intent else "")

        # Exit styles
        if otype == "EXIT":
            if regime == "CRISIS":
                return "DEFENSIVE_EXIT"
            if urgency == "HIGH":
                return "AGGRESSIVE"
            return "DEFENSIVE_EXIT"

        # Entry / Increase styles
        if otype in ("ENTRY", "INCREASE", "FLIP"):
            if regime == "CRISIS":
                return "NO_TRADE"  # Don't add risk in crisis
            if regime == "TENSION":
                return "SCALED_ENTRY"
            if urgency == "HIGH":
                return "AGGRESSIVE"
            if urgency == "MEDIUM":
                return "TWAP"
            if urgency == "LOW":
                return "DEFERRED"
            return "PASSIVE"

        # Decrease
        if otype == "DECREASE":
            if urgency == "HIGH":
                return "AGGRESSIVE"
            return "PASSIVE"

        # Default
        if urgency == "HIGH":
            return "AGGRESSIVE"
        return "PASSIVE"

    # -----------------------------------------------------------------
    # 5. Partial Fill Simulation
    # -----------------------------------------------------------------

    def simulate_fills(
        self, intents: List[OrderIntent], prices: Optional[pd.DataFrame] = None
    ) -> List[OrderIntent]:
        """
        Simulate partial fills for each OrderIntent based on execution style.

        Updates fill_status, filled_pct, and fill_price on each intent.
        """
        for intent in intents:
            if not intent.allowed:
                intent.fill_status = "REJECTED"
                intent.filled_pct = 0.0
                continue

            profile = _SLIPPAGE_PROFILES.get(
                intent.execution_style, _SLIPPAGE_PROFILES["PASSIVE"]
            )
            fill_pct = profile["fill_pct"]
            slip_bps = profile["slippage_bps"]

            if fill_pct <= 0:
                intent.fill_status = "DEFERRED"
                intent.filled_pct = 0.0
                continue

            # Get reference price
            ref_price = 0.0
            if prices is not None and intent.ticker in prices.columns:
                try:
                    ref_price = float(prices[intent.ticker].dropna().iloc[-1])
                except Exception:
                    pass

            if ref_price <= 0:
                ref_price = 100.0  # fallback

            intent.decision_price = ref_price
            intent.arrival_price = ref_price

            # Apply slippage
            slip_mult = slip_bps / 10_000.0
            if intent.side == "BUY":
                intent.fill_price = round(ref_price * (1.0 + slip_mult), 4)
            else:
                intent.fill_price = round(ref_price * (1.0 - slip_mult), 4)

            intent.filled_pct = round(fill_pct, 4)
            intent.expected_cost_bps = slip_bps

            if fill_pct >= 1.0:
                intent.fill_status = "FILLED"
            elif fill_pct > 0:
                intent.fill_status = "PARTIAL"
            else:
                intent.fill_status = "DEFERRED"

        filled_count = sum(1 for i in intents if i.fill_status == "FILLED")
        partial_count = sum(1 for i in intents if i.fill_status == "PARTIAL")
        log.info(
            "Simulated fills: %d filled, %d partial, %d deferred/rejected out of %d",
            filled_count, partial_count, len(intents) - filled_count - partial_count,
            len(intents),
        )
        return intents

    # -----------------------------------------------------------------
    # 6. Pending Order Manager
    # -----------------------------------------------------------------

    def manage_pending_orders(
        self, new_intents: List[OrderIntent]
    ) -> List[OrderIntent]:
        """
        Merge new intents with carried-forward pending orders.

        Removes stale orders (> MAX_PENDING_CYCLES old), avoids duplicates
        by ticker, and returns a combined list.
        """
        # Age existing pending orders
        carried: List[OrderIntent] = []
        for pending in self._pending_orders:
            pending.cycle_age += 1
            if pending.cycle_age > _MAX_PENDING_CYCLES:
                log.info("Stale pending order removed: %s (age=%d)", pending.ticker, pending.cycle_age)
                continue
            carried.append(pending)

        # Build set of tickers in new intents
        new_tickers = {i.ticker for i in new_intents}

        # Keep carried orders only for tickers NOT in new intents
        merged = [c for c in carried if c.ticker not in new_tickers]
        merged.extend(new_intents)

        # Separate out the ones that need to stay pending
        still_pending: List[OrderIntent] = []
        active: List[OrderIntent] = []
        for intent in merged:
            if intent.fill_status in ("DEFERRED", "PARTIAL") and intent.filled_pct < 1.0:
                still_pending.append(intent)
            active.append(intent)

        self._pending_orders = still_pending
        log.info(
            "Pending order manager: %d new, %d carried, %d merged, %d still pending",
            len(new_intents), len(carried), len(active), len(still_pending),
        )
        return active

    # -----------------------------------------------------------------
    # 7. Enhanced Exit Engine
    # -----------------------------------------------------------------

    def generate_exit_intents(
        self, inputs: Dict[str, Any], prices: Optional[pd.DataFrame] = None
    ) -> List[OrderIntent]:
        """
        Generate prioritized exit OrderIntents.

        Exit types (priority order):
        1. EMERGENCY — Risk Guardian emergency_unwind
        2. RISK_FORCED — must_reduce from Risk Guardian
        3. DECAY_FORCED — STRUCTURAL_DECAY / DEAD sleeves
        4. REGIME_FORCED — blocked in current regime
        5. TARGET_REBALANCE — target weight reduced to zero
        6. PROFIT_TAKING — above configurable threshold
        7. STOP_LOSS — below configurable threshold
        8. TIME_STOP — held > configured days
        """
        exit_intents: List[OrderIntent] = []

        rg = inputs.get("risk_guardian") or {}
        veto = rg.get("veto", {})
        emergency_unwind = veto.get("emergency_unwind_required", False)
        must_reduce = veto.get("must_reduce_existing_risk", False)
        vetoed_sleeves = set(veto.get("vetoed_sleeves", []))

        decay_data = inputs.get("alpha_decay") or {}
        decay_level = decay_data.get("decay_level", "HEALTHY")

        regime_data = inputs.get("regime_forecast") or {}
        regime = regime_data.get("predicted_regime", "NORMAL")

        pc = inputs.get("portfolio_construction") or {}
        target_weights = pc.get("weights", {})

        # Load current positions
        positions: List[dict] = []
        try:
            portfolio_path = ROOT / "data" / "paper_portfolio.json"
            if portfolio_path.exists():
                pdata = json.loads(portfolio_path.read_text(encoding="utf-8"))
                positions = pdata.get("positions", [])
        except Exception as exc:
            log.warning("Failed to load positions for exit engine: %s", exc)

        for pos in positions:
            ticker = pos.get("ticker", "")
            entry_price = float(pos.get("entry_price", 0))
            current_price = float(pos.get("current_price", entry_price))
            days_held = int(pos.get("days_held", 0))
            notional = float(pos.get("notional", 0))
            direction = pos.get("direction", "LONG")
            direction_sign = 1.0 if direction == "LONG" else -1.0

            if entry_price > 0:
                pnl_pct = direction_sign * (current_price - entry_price) / entry_price
            else:
                pnl_pct = 0.0

            exit_reason = ""
            urgency = "MEDIUM"
            execution_style = "DEFENSIVE_EXIT"

            # Priority 1: EMERGENCY
            if emergency_unwind:
                exit_reason = "EMERGENCY"
                urgency = "HIGH"
                execution_style = "EMERGENCY_EXIT"
            # Priority 2: RISK_FORCED
            elif must_reduce:
                exit_reason = "RISK_FORCED"
                urgency = "HIGH"
                execution_style = "AGGRESSIVE"
            # Priority 3: DECAY_FORCED
            elif decay_level in ("STRUCTURAL_DECAY", "DEAD"):
                exit_reason = "DECAY_FORCED"
                urgency = "HIGH"
                execution_style = "DEFENSIVE_EXIT"
            # Priority 4: REGIME_FORCED
            elif regime == "CRISIS" and ticker not in target_weights:
                exit_reason = "REGIME_FORCED"
                urgency = "HIGH"
                execution_style = "DEFENSIVE_EXIT"
            # Priority 5: TARGET_REBALANCE (target went to zero)
            elif ticker not in target_weights or target_weights.get(ticker, 0) == 0:
                exit_reason = "TARGET_REBALANCE"
                urgency = "MEDIUM"
                execution_style = "PASSIVE"
            # Priority 6: PROFIT_TAKING
            elif pnl_pct >= _PROFIT_TAKE_PCT:
                exit_reason = "PROFIT_TAKING"
                urgency = "MEDIUM"
                execution_style = "PASSIVE"
            # Priority 7: STOP_LOSS
            elif pnl_pct <= _STOP_LOSS_PCT:
                exit_reason = "STOP_LOSS"
                urgency = "HIGH"
                execution_style = "AGGRESSIVE"
            # Priority 8: TIME_STOP
            elif days_held >= _TIME_STOP_DAYS:
                exit_reason = "TIME_STOP"
                urgency = "MEDIUM"
                execution_style = "PASSIVE"

            if not exit_reason:
                continue

            intent = OrderIntent(
                order_id=f"EXIT_{ticker}_{uuid.uuid4().hex[:8]}",
                ticker=ticker,
                side="SELL" if direction == "LONG" else "BUY",
                order_type="EXIT",
                target_weight=0.0,
                current_weight=notional / _INITIAL_CAPITAL if _INITIAL_CAPITAL > 0 else 0.0,
                delta_notional=round(notional, 2),
                urgency=urgency,
                execution_style=execution_style,
                allowed=True,
                restrictions=[],
                expected_cost_bps=_SLIPPAGE_PROFILES.get(execution_style, {}).get("slippage_bps", 5.0),
                expected_impact_bps=0.0,
                sleeve_role="CORE",
                health_state=decay_level,
                regime_eligible=True,
                fill_status="PENDING",
                filled_pct=0.0,
                reason=exit_reason,
                decision_price=entry_price,
                arrival_price=current_price,
            )
            exit_intents.append(intent)

        # Sort by priority (EMERGENCY first)
        priority_order = {
            "EMERGENCY": 0, "RISK_FORCED": 1, "DECAY_FORCED": 2,
            "REGIME_FORCED": 3, "TARGET_REBALANCE": 4, "PROFIT_TAKING": 5,
            "STOP_LOSS": 6, "TIME_STOP": 7,
        }
        exit_intents.sort(key=lambda i: priority_order.get(i.reason, 99))

        log.info("Generated %d exit intents", len(exit_intents))
        return exit_intents

    # -----------------------------------------------------------------
    # 8. TCA Engine
    # -----------------------------------------------------------------

    def compute_tca(self, intents: List[OrderIntent]) -> Dict[str, Any]:
        """
        Compute Transaction Cost Analysis for filled orders.

        Returns per-order and summary TCA metrics.
        """
        tca: Dict[str, Any] = {
            "per_order": [],
            "summary": {
                "avg_slippage_bps": 0.0,
                "avg_shortfall_bps": 0.0,
                "total_cost_bps": 0.0,
                "by_side": {"BUY": [], "SELL": []},
                "by_regime": {},
                "by_style": {},
            },
        }

        filled = [i for i in intents if i.fill_status in ("FILLED", "PARTIAL") and i.filled_pct > 0]
        if not filled:
            return tca

        slippages: List[float] = []
        shortfalls: List[float] = []
        commission_bps = 1.0  # 1bps commission assumption

        for intent in filled:
            decision_px = intent.decision_price if intent.decision_price > 0 else intent.arrival_price
            arrival_px = intent.arrival_price if intent.arrival_price > 0 else decision_px
            fill_px = intent.fill_price if intent.fill_price > 0 else arrival_px

            if decision_px <= 0:
                continue

            # Slippage: fill vs arrival
            if intent.side == "BUY":
                slippage_bps = (fill_px / arrival_px - 1.0) * 10_000 if arrival_px > 0 else 0.0
                shortfall_bps = (fill_px / decision_px - 1.0) * 10_000 if decision_px > 0 else 0.0
            else:
                slippage_bps = (1.0 - fill_px / arrival_px) * 10_000 if arrival_px > 0 else 0.0
                shortfall_bps = (1.0 - fill_px / decision_px) * 10_000 if decision_px > 0 else 0.0

            market_impact_bps = abs(slippage_bps) * 0.6

            order_tca = {
                "order_id": intent.order_id,
                "ticker": intent.ticker,
                "side": intent.side,
                "execution_style": intent.execution_style,
                "decision_price": decision_px,
                "arrival_price": arrival_px,
                "fill_price": fill_px,
                "slippage_bps": round(slippage_bps, 2),
                "implementation_shortfall_bps": round(shortfall_bps, 2),
                "market_impact_bps": round(market_impact_bps, 2),
                "commission_bps": commission_bps,
                "total_cost_bps": round(abs(slippage_bps) + commission_bps, 2),
            }
            tca["per_order"].append(order_tca)

            slippages.append(abs(slippage_bps))
            shortfalls.append(abs(shortfall_bps))

            # Accumulate by side
            tca["summary"]["by_side"].setdefault(intent.side, []).append(abs(slippage_bps))

            # By style
            tca["summary"]["by_style"].setdefault(intent.execution_style, []).append(abs(slippage_bps))

        if slippages:
            tca["summary"]["avg_slippage_bps"] = round(float(np.mean(slippages)), 2)
            tca["summary"]["avg_shortfall_bps"] = round(float(np.mean(shortfalls)), 2)
            tca["summary"]["total_cost_bps"] = round(
                float(np.mean(slippages)) + commission_bps, 2
            )

        # Collapse by_side / by_style to averages
        for side_key in ("BUY", "SELL"):
            vals = tca["summary"]["by_side"].get(side_key, [])
            tca["summary"]["by_side"][side_key] = round(float(np.mean(vals)), 2) if vals else 0.0

        for style_key, vals in list(tca["summary"]["by_style"].items()):
            tca["summary"]["by_style"][style_key] = round(float(np.mean(vals)), 2) if vals else 0.0

        log.info(
            "TCA: %d orders, avg_slip=%.2fbps, avg_shortfall=%.2fbps",
            len(filled), tca["summary"]["avg_slippage_bps"],
            tca["summary"]["avg_shortfall_bps"],
        )
        return tca

    # -----------------------------------------------------------------
    # 9. Target vs Actual Reconciliation
    # -----------------------------------------------------------------

    def reconcile_target_vs_actual(
        self, layers: Dict[str, Any], filled_intents: List[OrderIntent]
    ) -> Dict[str, Any]:
        """
        Reconcile target weights vs actual post-execution weights.

        For each ticker computes drift and categorizes it.
        """
        recon: Dict[str, Any] = {"positions": {}, "summary": {}}

        desired = layers.get("desired_portfolio", {})
        realized = layers.get("realized_portfolio", {})
        executable = layers.get("executable_portfolio", {})

        # Build filled tickers set
        filled_tickers = {i.ticker for i in filled_intents if i.fill_status in ("FILLED", "PARTIAL")}
        rejected_tickers = {i.ticker for i in filled_intents if i.fill_status == "REJECTED"}

        # Build risk-blocked tickers
        risk_blocked: Dict[str, Any] = {}
        for ticker, row in layers.get("risk_approved_portfolio", {}).items():
            if row.get("order_type") == "HOLD" and desired.get(ticker, {}).get("order_type") != "HOLD":
                risk_blocked[ticker] = True

        all_tickers = set(desired.keys()) | set(realized.keys())

        total_drift = 0.0
        for ticker in all_tickers:
            target_w = desired.get(ticker, {}).get("target_weight", 0.0)
            actual_w = realized.get(ticker, {}).get("current_weight", 0.0)
            drift = abs(target_w - actual_w)
            total_drift += drift

            # Categorize
            if drift < _DUST_THRESHOLD_WEIGHT:
                category = "WITHIN_TOLERANCE"
                next_action = "HOLD"
            elif ticker in filled_tickers:
                category = "AWAITING_EXECUTION"
                next_action = "MONITOR_FILL"
            elif ticker in risk_blocked:
                category = "BLOCKED_BY_RISK"
                next_action = "WAIT_FOR_RISK_CLEARANCE"
            elif ticker in rejected_tickers:
                category = "BLOCKED_BY_DECAY"
                next_action = "REVIEW_ALPHA"
            else:
                category = "STALE_POSITION"
                next_action = "EVALUATE_EXIT"

            recon["positions"][ticker] = {
                "target_weight": round(target_w, 6),
                "actual_weight": round(actual_w, 6),
                "drift": round(drift, 6),
                "drift_category": category,
                "next_action": next_action,
            }

        recon["summary"] = {
            "total_drift": round(total_drift, 6),
            "n_within_tolerance": sum(
                1 for p in recon["positions"].values() if p["drift_category"] == "WITHIN_TOLERANCE"
            ),
            "n_awaiting": sum(
                1 for p in recon["positions"].values() if p["drift_category"] == "AWAITING_EXECUTION"
            ),
            "n_blocked": sum(
                1 for p in recon["positions"].values()
                if p["drift_category"] in ("BLOCKED_BY_RISK", "BLOCKED_BY_DECAY")
            ),
            "n_stale": sum(
                1 for p in recon["positions"].values() if p["drift_category"] == "STALE_POSITION"
            ),
        }

        log.info(
            "Reconciliation: drift=%.4f, within_tol=%d, awaiting=%d, blocked=%d, stale=%d",
            total_drift,
            recon["summary"]["n_within_tolerance"],
            recon["summary"]["n_awaiting"],
            recon["summary"]["n_blocked"],
            recon["summary"]["n_stale"],
        )
        return recon

    # -----------------------------------------------------------------
    # 10. Execution Governance
    # -----------------------------------------------------------------

    def apply_execution_governance(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute execution governance state from all upstream inputs.

        Determines what the execution desk is allowed to do this cycle.
        """
        rg = inputs.get("risk_guardian") or {}
        veto = rg.get("veto", {})
        regime_data = inputs.get("regime_forecast") or {}
        regime = regime_data.get("predicted_regime", "NORMAL")
        decay_data = inputs.get("alpha_decay") or {}
        decay_level = decay_data.get("decay_level", "HEALTHY")

        emergency_unwind = veto.get("emergency_unwind_required", False)
        must_reduce = veto.get("must_reduce_existing_risk", False)
        can_add_new = veto.get("can_allocate_new_risk", True)
        can_execute = veto.get("can_execute_new_trades", True)
        vetoed_sleeves = veto.get("vetoed_sleeves", [])

        # Regime caution
        regime_caution_map = {
            "CALM": "NORMAL",
            "NORMAL": "NORMAL",
            "TENSION": "CAUTIOUS",
            "CRISIS": "DEFENSIVE",
        }
        regime_caution = regime_caution_map.get(regime, "NORMAL")

        if emergency_unwind:
            regime_caution = "HALT"

        # Conservative defaults when Risk Guardian is missing
        has_risk_guardian = "risk_guardian" in inputs.get("available_inputs", [])
        if not has_risk_guardian:
            can_add_new = False  # conservative

        # Blocked sleeves: from veto + decay
        blocked_sleeves = list(set(vetoed_sleeves))
        if decay_level in ("STRUCTURAL_DECAY", "DEAD"):
            blocked_sleeves.append(f"ALL_ENTRY_BLOCKED_DECAY_{decay_level}")

        # Max turnover budget
        turnover_budget_map = {
            "CALM": 0.30, "NORMAL": 0.25, "TENSION": 0.15, "CRISIS": 0.05,
        }
        max_turnover = turnover_budget_map.get(regime, 0.20)
        if must_reduce:
            max_turnover = min(max_turnover, 0.10)

        governance = {
            "can_add_new_risk": can_add_new and not emergency_unwind,
            "can_reduce_risk": True,  # always allowed
            "can_rebalance_normally": can_execute and not emergency_unwind and not must_reduce,
            "emergency_mode": emergency_unwind,
            "regime_caution": regime_caution,
            "blocked_sleeves": blocked_sleeves,
            "max_turnover_budget": round(max_turnover, 4),
        }

        log.info(
            "Execution governance: add_risk=%s, reduce=%s, rebalance=%s, "
            "emergency=%s, caution=%s, turnover_budget=%.2f",
            governance["can_add_new_risk"], governance["can_reduce_risk"],
            governance["can_rebalance_normally"], governance["emergency_mode"],
            governance["regime_caution"], governance["max_turnover_budget"],
        )
        return governance

    # -----------------------------------------------------------------
    # 11. Machine Summary
    # -----------------------------------------------------------------

    def build_machine_summary(
        self,
        governance: Dict[str, Any],
        intents: List[OrderIntent],
        tca: Dict[str, Any],
        recon: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build machine-readable summary for downstream agents.
        """
        regime_data = inputs.get("regime_forecast") or {}
        regime = regime_data.get("predicted_regime", "NORMAL")
        transition_prob = regime_data.get("transition_probability", 0.0)
        if transition_prob > 0.6:
            transition_state = "ACTIVE_TRANSITION"
        elif transition_prob > 0.4:
            transition_state = "EARLY_WARNING"
        else:
            transition_state = "STABLE"

        target_count = sum(1 for i in intents if i.allowed)
        executed_count = sum(1 for i in intents if i.fill_status in ("FILLED", "PARTIAL"))
        pending_count = sum(1 for i in intents if i.fill_status in ("DEFERRED", "PARTIAL"))
        rejected_count = sum(1 for i in intents if i.fill_status == "REJECTED")

        turnover_executed = sum(
            abs(i.delta_notional) * i.filled_pct
            for i in intents if i.fill_status in ("FILLED", "PARTIAL")
        ) / _INITIAL_CAPITAL if _INITIAL_CAPITAL > 0 else 0.0

        # Top blockers
        top_blockers = [
            f"{i.ticker} {'; '.join(i.restrictions)}"
            for i in intents if not i.allowed and i.restrictions
        ][:5]

        # Top forced exits
        top_forced_exits = [
            f"{i.ticker} {i.reason}"
            for i in intents if i.order_type == "EXIT" and i.reason in (
                "EMERGENCY", "RISK_FORCED", "DECAY_FORCED", "REGIME_FORCED"
            )
        ][:5]

        # Top pending
        top_pending = [
            f"{i.ticker} partial fill {i.filled_pct:.0%}"
            for i in intents if i.fill_status == "PARTIAL"
        ][:5]

        summary = {
            "can_add_new_risk": governance.get("can_add_new_risk", False),
            "can_reduce_risk": governance.get("can_reduce_risk", True),
            "emergency_mode": governance.get("emergency_mode", False),
            "target_positions": target_count,
            "executed_positions": executed_count,
            "pending_count": pending_count,
            "rejected_count": rejected_count,
            "avg_slippage_bps": tca.get("summary", {}).get("avg_slippage_bps", 0.0),
            "implementation_shortfall_bps": tca.get("summary", {}).get("avg_shortfall_bps", 0.0),
            "turnover_executed": round(turnover_executed, 4),
            "drift_to_target": recon.get("summary", {}).get("total_drift", 0.0),
            "active_regime": regime,
            "transition_state": transition_state,
            "execution_caution_level": governance.get("regime_caution", "NORMAL"),
            "top_blockers": top_blockers,
            "top_forced_exits": top_forced_exits,
            "top_pending": top_pending,
        }
        return summary

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

        # Step 8: Advanced execution analytics
        try:
            reconciliation = self.reconcile_portfolio()
            summary["reconciliation"] = reconciliation
            log.info("Reconciliation: %s", reconciliation.get("detail", ""))
        except Exception as exc:
            log.warning("Reconciliation skipped: %s", exc)

        try:
            exec_quality = self.compute_execution_quality()
            summary["execution_quality"] = exec_quality
            log.info("Execution quality: %s", exec_quality.get("detail", ""))
        except Exception as exc:
            log.warning("Execution quality skipped: %s", exc)

        # =============================================================
        # INSTITUTIONAL EXECUTION DESK — steps 8b..14
        # Wrapped in try/except to preserve existing behaviour on failure
        # =============================================================
        institutional_summary: Dict[str, Any] = {}
        all_intents: List[OrderIntent] = []
        try:
            # 8b. Assemble all upstream inputs
            exec_inputs = self.assemble_execution_inputs()
            summary["execution_inputs_available"] = exec_inputs.get("available_inputs", [])

            # 8c. Apply execution governance
            governance = self.apply_execution_governance(exec_inputs)
            summary["execution_governance"] = governance

            # 8d. Compute portfolio layers
            layers = self.compute_portfolio_layers(exec_inputs)
            summary["portfolio_layers_computed"] = True

            # 8e. Generate exit intents (prioritized)
            exit_intents = self.generate_exit_intents(exec_inputs, prices)

            # 8f. Generate entry / rebalance intents
            entry_intents = self.build_order_intents(layers, exec_inputs)

            # 8g. Merge with pending orders
            combined_intents = exit_intents + entry_intents
            all_intents = self.manage_pending_orders(combined_intents)

            # 8h. Simulate fills (partial)
            all_intents = self.simulate_fills(all_intents, prices)

            # 8i. Reconcile target vs actual
            recon = self.reconcile_target_vs_actual(layers, all_intents)
            summary["target_vs_actual"] = recon

            # 8j. Compute TCA
            tca = self.compute_tca(all_intents)
            summary["tca"] = tca

            # 8k. Build machine summary
            institutional_summary = self.build_machine_summary(
                governance, all_intents, tca, recon, exec_inputs
            )
            summary["machine_summary"] = institutional_summary

            # 8l. Save institutional execution state
            try:
                inst_out_path = Path(__file__).resolve().parent / "execution_state.json"
                inst_out_path.write_text(
                    json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "machine_summary": institutional_summary,
                        "governance": governance,
                        "tca_summary": tca.get("summary", {}),
                        "reconciliation_summary": recon.get("summary", {}),
                        "order_intents": [i.to_dict() for i in all_intents],
                        "pending_count": len(self._pending_orders),
                    }, indent=2, default=str, ensure_ascii=False),
                    encoding="utf-8",
                )
                log.info("Saved institutional execution state to %s", inst_out_path)
            except Exception as exc:
                log.warning("Failed to save execution_state.json: %s", exc)

            log.info(
                "Institutional desk: %d intents, %d filled, %d pending, %d rejected",
                len(all_intents),
                sum(1 for i in all_intents if i.fill_status in ("FILLED", "PARTIAL")),
                sum(1 for i in all_intents if i.fill_status in ("DEFERRED", "PARTIAL")),
                sum(1 for i in all_intents if i.fill_status == "REJECTED"),
            )

        except Exception as exc:
            log.error("Institutional execution desk failed (legacy path intact): %s", exc)
            summary["institutional_error"] = str(exc)

        # Step 9: Save execution log
        self._save_execution_log()

        # Step 10: Publish to bus
        if _IMPORTS_OK.get("agent_bus"):
            try:
                bus = get_bus()
                bus_payload: Dict[str, Any] = {
                    "date": today,
                    "entries": len(summary["entries"]),
                    "exits": len(summary["exits"]),
                    "signals_found": summary["signals_found"],
                    "signals_sized": summary["signals_sized"],
                    "risk_guardian_ok": summary["risk_guardian_ok"],
                }
                if institutional_summary:
                    bus_payload["machine_summary"] = institutional_summary
                bus.publish("execution_agent", bus_payload)
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
